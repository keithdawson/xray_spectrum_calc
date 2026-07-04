import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import io
import json
from apply_filter_and_fluorescence import run_stack, PRACTICAL_FILTER_MATERIALS, SYM_TO_Z
from bundled_data import DEFAULT_SPECTRA

st.set_page_config(page_title='X-ray Spectrum Calculator', layout='wide')

st.title("⚛️ X-ray Spectrum Attenuation & Fluorescence")

attenuation_file = 'attenuation_coefficients_long.csv'
edpl_folder = 'pyne_edpl'

CONFIG_FILE = '.last_filters.json'

def load_filters():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for f_item in data:
                        f_item['thickness'] = max(0.0, float(f_item.get('thickness', 0.0)))
                    return data
        except Exception:
            pass
    return []


def save_filters():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(st.session_state.filters, f)
    except Exception:
        pass

# Initialize session state variables
if 'filters' not in st.session_state:
    st.session_state.filters = load_filters()
if 'calculated_spectrum' not in st.session_state:
    st.session_state.calculated_spectrum = None
if 'last_calculated_params' not in st.session_state:
    st.session_state.last_calculated_params = None
if 'fluor_opt_results' not in st.session_state:
    st.session_state.fluor_opt_results = None
if 'filter_opt_results' not in st.session_state:
    st.session_state.filter_opt_results = None

MATERIALS_CONFIG_FILE = '.material_settings.json'
DEFAULT_ENABLED_MATERIALS = ['Al', 'Fe', 'Cu', 'Mo', 'Ag', 'Sn', 'W', 'Au', 'Pb']


def load_enabled_materials():
    if os.path.exists(MATERIALS_CONFIG_FILE):
        try:
            with open(MATERIALS_CONFIG_FILE, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                valid = [m for m in data if m in PRACTICAL_FILTER_MATERIALS]
                if valid:
                    return valid
        except Exception:
            pass
    return DEFAULT_ENABLED_MATERIALS.copy()


def save_enabled_materials(mats):
    try:
        with open(MATERIALS_CONFIG_FILE, 'w') as f:
            json.dump(mats, f)
    except Exception:
        pass


if 'enabled_materials' not in st.session_state:
    st.session_state.enabled_materials = load_enabled_materials()


def render_diagram(dist):
    svg = f'''<svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="background-color: #f8f9fa; border-radius: 10px; border: 1px solid #dee2e6;">
    <defs>
        <marker id="arrow-end" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 Z" fill="#6c757d" />
        </marker>
        <marker id="arrow-start" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
            <path d="M10,0 L0,5 L10,10 Z" fill="#6c757d" />
        </marker>
    </defs>

    <polygon points="50,100 545,48 545,152" fill="rgba(0, 123, 255, 0.1)" stroke="rgba(0, 123, 255, 0.5)" stroke-width="1" />

    <circle cx="50" cy="100" r="10" fill="#343a40" />
    <text x="50" y="130" text-anchor="middle" font-family="Arial" font-size="12" fill="#495057">Source</text>

    <rect x="100" y="83" width="9" height="34" fill="#adb5bd" />
    <rect x="114" y="83" width="9" height="34" fill="#adb5bd" />
    <text x="111" y="70" text-anchor="middle" font-family="Arial" font-size="12" fill="#495057">Filters</text>

    <rect x="316" y="58" width="8" height="84" fill="#1e7145" rx="1" />
    <rect x="316" y="66" width="8" height="7" fill="#ffd700" />
    <rect x="316" y="83" width="8" height="7" fill="#ffd700" />
    <rect x="316" y="100" width="8" height="7" fill="#ffd700" />
    <rect x="316" y="117" width="8" height="7" fill="#ffd700" />
    <text x="320" y="48" text-anchor="middle" font-family="Arial" font-size="12" fill="#495057">PCB DUT</text>

    <rect x="545" y="42" width="16" height="116" fill="#212529" rx="2" />
    <rect x="548" y="46" width="4" height="108" fill="#5c6bc0" />
    <text x="553" y="172" text-anchor="middle" font-family="Arial" font-size="12" fill="#495057">Detector</text>

    <line x1="50" y1="182" x2="316" y2="182" stroke="#6c757d" stroke-width="2" marker-start="url(#arrow-start)" marker-end="url(#arrow-end)" />
    <text x="183" y="172" text-anchor="middle" font-family="Arial" font-weight="bold" font-size="13" fill="#212529">Distance: {dist:.1f} mm</text>
</svg>'''
    return " ".join(line.strip() for line in svg.splitlines())

# --- Global Settings & Execution Header ---
with st.container():
    head_c1, head_c2 = st.columns([4, 1])
    head_c1.subheader('Global Settings & Context')
    with head_c2.popover('⚙️ Materials'):
        st.caption('Filter materials offered in the Filter Stack dropdowns and swept by the Cascading Fluorescence and CNR Filter Optimizers. All options are elements practically available as thin foils or discs. More selections make the optimizer sweeps take longer.')
        mat_cols = st.columns(6)
        selected_mats = []
        for i, sym in enumerate(sorted(PRACTICAL_FILTER_MATERIALS, key=lambda s: SYM_TO_Z[s])):
            checked = mat_cols[i % 6].checkbox(
                f'{sym} ({SYM_TO_Z[sym]})',
                value=sym in st.session_state.enabled_materials,
                key=f'mat_enable_{sym}'
            )
            if checked:
                selected_mats.append(sym)
        if not selected_mats:
            st.warning('No materials selected — falling back to the default set.')
            selected_mats = DEFAULT_ENABLED_MATERIALS.copy()
        if selected_mats != st.session_state.enabled_materials:
            st.session_state.enabled_materials = selected_mats
            save_enabled_materials(selected_mats)

    SORTED_MATERIALS = ['Air'] + sorted(st.session_state.enabled_materials, key=lambda s: SYM_TO_Z[s])

    import ct_materials
    if 'scint_thick_input' not in st.session_state:
        st.session_state.scint_thick_input = 0.14
        
    def on_scint_mat_change():
        if st.session_state.scint_mat_select == 'Gadox':
            st.session_state.scint_thick_input = 0.14
        elif st.session_state.scint_mat_select == 'CsI':
            st.session_state.scint_thick_input = 0.55

    top_c1, top_c2, top_c3 = st.columns([1.2, 2.0, 1.0])
    
    with top_c1:
        st.markdown("**Beam Basics**")
        bb_c1, bb_c2 = st.columns(2)
        with bb_c1:
            beam_angle = st.slider('Beam Cone Angle (deg)', 15.0, 120.0, 30.0, step=2.0)
        with bb_c2:
            tube_window = st.selectbox('Tube Window', options=['Beryllium (2mm)', 'Aluminum (1mm)', 'None'], index=1)
            distance_to_target = st.number_input('Distance to Target (mm)', value=1000.0, min_value=0.0, step=10.0, format='%.1f')
        
    with top_c2:
        st.markdown("**Target & Detector Context**")
        ctx_c1, ctx_c2, ctx_c3 = st.columns(3)
        with ctx_c1:
            target_a_mat = st.selectbox('Target A (Background)', options=ct_materials.CT_MATERIALS, index=0)
            target_a_thick = st.number_input('Thickness A (mm)', value=1.0, min_value=0.0, format='%.2f', step=0.1)
        with ctx_c2:
            target_b_mat = st.selectbox('Target B (Feature)', options=ct_materials.CT_MATERIALS, index=1)
            target_b_thick = st.number_input('Thickness B (mm)', value=0.035, min_value=0.0, format='%.3f', step=0.01)
        with ctx_c3:
            scint_mat = st.selectbox('Detector Scintillator', options=ct_materials.SCINTILLATORS, index=0, key='scint_mat_select', on_change=on_scint_mat_change)
            scint_thick = st.number_input('Scint Thickness (mm)', key='scint_thick_input', min_value=0.0, format='%.2f', step=0.01)
            
    with top_c3:
        st.markdown("**Execution**")
        st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
        calculate_btn = st.button('🚀 Calculate Spectrum', type='primary', width='stretch')
        st.caption("Press to calculate raw spectrum and contrast metrics.")

st.markdown('---')

col1, col2 = st.columns([1, 1.8])

with col1:
    st.subheader('Input Configuration')
    
    st.markdown(render_diagram(distance_to_target), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    local_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'attenuation' not in f.lower()]
    all_options = ['Generate with SpekPy', 'Upload new file...'] + list(DEFAULT_SPECTRA.keys()) + [f for f in local_files if f not in DEFAULT_SPECTRA]
    
    default_idx = 0
    if 'Generate with SpekPy' in all_options:
        default_idx = all_options.index('Generate with SpekPy')
        
    input_option = st.selectbox(
        'Select Input Spectrum',
        options=all_options,
        index=default_idx
    )
    
    uploaded_file = None
    spectrum_path = None
    embedded_content = None
    
    spekpy_kvp = 220.0
    spekpy_th = 22.0
    spekpy_targ = 'W'
    spekpy_dk = 0.1

    if input_option == 'Generate with SpekPy':
        st.markdown('**SpekPy Generation Settings**')
        sp_cols = st.columns(4)
        spekpy_kvp = sp_cols[0].number_input('Tube Voltage (kVp)', min_value=20.0, max_value=300.0, value=220.0, step=1.0)
        spekpy_th = sp_cols[1].number_input('Target Angle (deg)', min_value=5.0, max_value=30.0, value=22.0, step=1.0)
        spekpy_targ = sp_cols[2].selectbox('Anode Material', options=['W', 'Mo', 'Rh'], index=0)
        spekpy_dk = sp_cols[3].selectbox('Energy Bin (keV)', options=[0.1, 0.2, 0.5], index=0, help="Spectrum energy resolution. Finer bins sharpen characteristic lines and fluorescence peaks at a small speed cost.")
    elif input_option == 'Upload new file...':
        uploaded_file = st.file_uploader('Upload Input Spectrum CSV', type=['csv'])
    elif input_option in DEFAULT_SPECTRA:
        embedded_content = DEFAULT_SPECTRA[input_option]
        spectrum_path = input_option
    else:
        spectrum_path = input_option

    st.markdown('### Filter Stack')
    st.caption('*Note: Filters are applied sequentially. Filter 1 is physically closest to the X-ray source.*')
    if 'filters' not in st.session_state:
        st.session_state.filters = load_filters()
    
    def sync_filter_widget_state():
        for k in list(st.session_state.keys()):
            if k.startswith('mat_sel_') or k.startswith('mat_custom_') or k.startswith('thick_'):
                del st.session_state[k]
        for i, f in enumerate(st.session_state.filters):
            st.session_state[f'mat_sel_{i}'] = f['material'] if f['material'] in SORTED_MATERIALS else 'Custom'
            if f['material'] not in SORTED_MATERIALS:
                st.session_state[f'mat_custom_{i}'] = f['material']
            st.session_state[f'thick_{i}'] = f['thickness']

    def add_filter():
        if len(st.session_state.filters) > 0:
            prev = st.session_state.filters[-1]
            st.session_state.filters.append({'material': prev['material'], 'thickness': prev['thickness']})
        else:
            st.session_state.filters.append({'material': 'Ag', 'thickness': 0.2})
        sync_filter_widget_state()
        save_filters()
        
    def remove_filter(index):
        st.session_state.filters.pop(index)
        sync_filter_widget_state()
        save_filters()

    def move_up(index):
        if index > 0:
            st.session_state.filters[index-1], st.session_state.filters[index] = st.session_state.filters[index], st.session_state.filters[index-1]
            sync_filter_widget_state()
            save_filters()

    def move_down(index):
        if index < len(st.session_state.filters) - 1:
            st.session_state.filters[index+1], st.session_state.filters[index] = st.session_state.filters[index], st.session_state.filters[index+1]
            sync_filter_widget_state()
            save_filters()

    for i, filter_item in enumerate(st.session_state.filters):
        cols = st.columns([4, 3, 1, 1, 1])
        
        mat_selection = cols[0].selectbox(
            f'Material {i+1}', 
            options=SORTED_MATERIALS + ['Custom'],
            index=SORTED_MATERIALS.index(filter_item['material']) if filter_item['material'] in SORTED_MATERIALS else len(SORTED_MATERIALS),
            key=f'mat_sel_{i}'
        )
        
        if mat_selection == 'Custom':
            material_val = cols[0].text_input(f'Custom Mat {i+1}', value=filter_item['material'], key=f'mat_custom_{i}')
        else:
            material_val = mat_selection
            
        filter_item['material'] = material_val
        filter_item['thickness'] = cols[1].number_input(f'Thick (mm) {i+1}', value=filter_item['thickness'], min_value=0.0, format='%.2f', step=0.1, key=f'thick_{i}')
        
        cols[2].markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        cols[2].button('↑', on_click=move_up, args=(i,), key=f'up_{i}', disabled=(i==0), help="Move up")
        
        cols[3].markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        cols[3].button('↓', on_click=move_down, args=(i,), key=f'down_{i}', disabled=(i==len(st.session_state.filters)-1), help="Move down")
        
        cols[4].markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        cols[4].button('❌', on_click=remove_filter, args=(i,), key=f'rem_{i}', help="Remove filter")

    st.button('➕ Add Filter', on_click=add_filter)
    save_filters()


with col2:
    st.subheader('Results')

    disp_c1, disp_c2, disp_c3 = st.columns([1.8, 1.0, 1.2])
    save_fluoro = disp_c1.checkbox('Show Secondary (Fluoro + Scatter)', value=True, key='save_fluoro')
    log_y = disp_c2.checkbox('Log Y-Axis', value=False, key='log_y_toggle')
    display_bin = disp_c3.selectbox('Chart Bin (keV)', options=[1.0, 0.5, 'Native'], index=0,
                                    label_visibility='collapsed',
                                    format_func=lambda v: f'Chart bin: {v} keV' if v != 'Native' else 'Chart bin: Native',
                                    help='Display-only rebinning. Calculations always run at the full spectrum resolution; wider chart bins tame the tall characteristic-line spikes so the continuum shape is readable.')

    def format_sci(val):
        if val == 0: return "0"
        s = f"{val:.4e}"
        if s.endswith("e+00") or s.endswith("e-00"): return s[:-4]
        return s.replace("e+", "e").replace("e-0", "e-").replace("e0", "e")

    def format_power(watts):
        if watts <= 0:
            return '0 W'
        for scale, unit in [(1.0, 'W'), (1e-3, 'mW'), (1e-6, 'µW'), (1e-9, 'nW'), (1e-12, 'pW')]:
            if watts >= scale:
                return f'{watts / scale:.3f} {unit}'
        return f'{watts:.3e} W'

    def rebin_display(E, phi, bin_kev):
        """Rebins a fluence density spectrum onto a coarser uniform grid for
        display only (area-preserving average). Returns inputs unchanged if
        they are already coarser than the requested bin."""
        E = np.asarray(E, dtype=float)
        phi = np.asarray(phi, dtype=float)
        if len(E) < 3 or np.median(np.gradient(E)) >= bin_kev:
            return E, phi
        edges = np.arange(E.min() - 1e-9, E.max() + bin_kev, bin_kev)
        integral, _ = np.histogram(E, bins=edges, weights=phi * np.gradient(E))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, integral / bin_kev

    def display_xy(x, y):
        if display_bin == 'Native':
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return rebin_display(x, y, float(display_bin))

    current_params = {
        'input_option': input_option,
        'spekpy_kvp': spekpy_kvp,
        'spekpy_th': spekpy_th,
        'spekpy_targ': spekpy_targ,
        'spekpy_dk': spekpy_dk,
        'tube_window': tube_window,
        'uploaded_file_name': uploaded_file.name if uploaded_file else None,
        'beam_angle': beam_angle,
        'distance_to_target': distance_to_target,
        'save_fluoro': save_fluoro,
        'filters': [f.copy() for f in st.session_state.filters]
    }
    
    run_calculation = False
    if calculate_btn:
        run_calculation = True
    elif st.session_state.calculated_spectrum is None:
        run_calculation = True
        
    if run_calculation:
        current_spectrum_path = None
        
        if input_option == 'Generate with SpekPy':
            import spekpy
            s = spekpy.Spek(kvp=spekpy_kvp, th=spekpy_th, targ=spekpy_targ, dk=spekpy_dk)
            if tube_window == 'Beryllium (2mm)':
                s.filter('Be', 2.0)
            elif tube_window == 'Aluminum (1mm)':
                s.filter('Al', 1.0)
            energy, fluence = s.get_spectrum()
            df_spek = pd.DataFrame({'Energy_KeV': energy, 'Photons_per_s': fluence})
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as tmp_in:
                df_spek.to_csv(tmp_in.name, index=False)
                current_spectrum_path = tmp_in.name
        elif embedded_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as tmp_in:
                tmp_in.write(embedded_content)
                current_spectrum_path = tmp_in.name
        elif spectrum_path:
            current_spectrum_path = spectrum_path
        elif uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_in:
                tmp_in.write(uploaded_file.getvalue())
                current_spectrum_path = tmp_in.name
        
        if current_spectrum_path:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    out_path = os.path.join(tmp_dir, 'out.csv')
                    final_filter_stack = [f.copy() for f in st.session_state.filters]
                    if distance_to_target > 0:
                        final_filter_stack.append({'material': 'Air', 'thickness': distance_to_target})
                    geom_params = {'beam_angle_deg': beam_angle}
                    
                    run_stack(
                        spectrum_csv=current_spectrum_path,
                        filter_stack=final_filter_stack,
                        att_file=attenuation_file,
                        edpl_folder=edpl_folder,
                        geom_params=geom_params,
                        out_path=out_path,
                        save_fluorescence=save_fluoro
                    )
                    
                    res_df = pd.read_csv(out_path)
                    input_df = pd.read_csv(current_spectrum_path)
                    
                    # Calculate Power at Target
                    E_out = res_df['Energy_KeV'].values
                    Phi_out = res_df['Photons_per_s'].values
                    dE_out = np.gradient(E_out)
                    power_watts = np.sum(Phi_out * dE_out * E_out * 1.60218e-16)
                    total_flux = np.sum(Phi_out * dE_out)
                    
                    delta_power_str = None
                    delta_flux_str = None
                    
                    if 'prev_power_watts' in st.session_state and st.session_state.prev_power_watts > 0:
                        pct_power = (power_watts - st.session_state.prev_power_watts) / st.session_state.prev_power_watts * 100.0
                        delta_power_str = f"{pct_power:.2f}%"
                        
                    if 'prev_total_flux' in st.session_state and st.session_state.prev_total_flux > 0:
                        pct_flux = (total_flux - st.session_state.prev_total_flux) / st.session_state.prev_total_flux * 100.0
                        delta_flux_str = f"{pct_flux:.2f}%"
                        
                    st.session_state.prev_power_watts = power_watts
                    st.session_state.prev_total_flux = total_flux
                    
                    fluoro_df = None
                    f_power = 0.0
                    f_flux = 0.0
                    pct_of_total = 0.0
                    f_delta_str = None
                    
                    if save_fluoro:
                        fluoro_path = out_path[:-4] + '_fluoro.csv'
                        if os.path.exists(fluoro_path):
                            fluoro_df = pd.read_csv(fluoro_path)
                            if not fluoro_df.empty:
                                f_E = fluoro_df['Energy_KeV'].values
                                f_Phi = fluoro_df['Photons_per_s'].values
                                f_dE = np.gradient(f_E)
                                f_power = np.sum(f_Phi * f_dE * f_E * 1.60218e-16)
                                f_flux = np.sum(f_Phi * f_dE)
                                pct_of_total = (f_power / power_watts) * 100.0 if power_watts > 0 else 0.0
                                
                                if 'prev_f_power' in st.session_state and st.session_state.prev_f_power > 0:
                                    pct_f = (f_power - st.session_state.prev_f_power) / st.session_state.prev_f_power * 100.0
                                    f_delta_str = f"{pct_f:.2f}%"
                                st.session_state.prev_f_power = f_power

                    scatter_df = None
                    s_power = 0.0
                    s_flux = 0.0
                    s_pct_of_total = 0.0
                    s_delta_str = None

                    if save_fluoro:
                        scatter_path = out_path[:-4] + '_scatter.csv'
                        if os.path.exists(scatter_path):
                            scatter_df = pd.read_csv(scatter_path)
                            if not scatter_df.empty:
                                s_E = scatter_df['Energy_KeV'].values
                                s_Phi = scatter_df['Photons_per_s'].values
                                s_dE = np.gradient(s_E)
                                s_power = np.sum(s_Phi * s_dE * s_E * 1.60218e-16)
                                s_flux = np.sum(s_Phi * s_dE)
                                s_pct_of_total = (s_power / power_watts) * 100.0 if power_watts > 0 else 0.0

                                if 'prev_s_power' in st.session_state and st.session_state.prev_s_power > 0:
                                    pct_s = (s_power - st.session_state.prev_s_power) / st.session_state.prev_s_power * 100.0
                                    s_delta_str = f"{pct_s:.2f}%"
                                st.session_state.prev_s_power = s_power

                    st.session_state.calculated_spectrum = {
                        'res_df': res_df,
                        'fluoro_df': fluoro_df,
                        'scatter_df': scatter_df,
                        's_power': s_power,
                        's_flux': s_flux,
                        's_pct_of_total': s_pct_of_total,
                        's_delta_str': s_delta_str,
                        'input_df': input_df,
                        'power_watts': power_watts,
                        'total_flux': total_flux,
                        'f_power': f_power,
                        'f_flux': f_flux,
                        'pct_of_total': pct_of_total,
                        'delta_power_str': delta_power_str,
                        'delta_flux_str': delta_flux_str,
                        'f_delta_str': f_delta_str
                    }
                    st.session_state.last_calculated_params = current_params
                    
                if not spectrum_path or (spectrum_path and spectrum_path in DEFAULT_SPECTRA):
                    if os.path.exists(current_spectrum_path):
                        os.remove(current_spectrum_path)
            except Exception as e:
                st.error(f'Calculation Error: {e}')
        else:
            st.warning('Please select a spectrum file or upload a new one.')

    # Render results from state if they exist
    if st.session_state.calculated_spectrum is not None:
        calc_data = st.session_state.calculated_spectrum
        res_df = calc_data['res_df']
        fluoro_df = calc_data['fluoro_df']
        scatter_df = calc_data.get('scatter_df')
        input_df = calc_data['input_df']
        power_watts = calc_data['power_watts']
        total_flux = calc_data['total_flux']
        f_power = calc_data['f_power']
        f_flux = calc_data['f_flux']
        pct_of_total = calc_data['pct_of_total']
        
        # Check for staleness
        is_stale = False
        if st.session_state.last_calculated_params is not None:
            p1 = current_params
            p2 = st.session_state.last_calculated_params
            if (p1['input_option'] != p2['input_option'] or
                p1['spekpy_kvp'] != p2['spekpy_kvp'] or
                p1['spekpy_th'] != p2['spekpy_th'] or
                p1['spekpy_targ'] != p2['spekpy_targ'] or
                p1['spekpy_dk'] != p2.get('spekpy_dk') or
                p1['tube_window'] != p2['tube_window'] or
                p1['uploaded_file_name'] != p2['uploaded_file_name'] or
                p1['beam_angle'] != p2['beam_angle'] or
                p1['distance_to_target'] != p2['distance_to_target'] or
                p1['save_fluoro'] != p2['save_fluoro'] or
                len(p1['filters']) != len(p2['filters'])):
                is_stale = True
            else:
                for i in range(len(p1['filters'])):
                    if (p1['filters'][i]['material'] != p2['filters'][i]['material'] or
                        p1['filters'][i]['thickness'] != p2['filters'][i]['thickness']):
                        is_stale = True
                        break
                        
        if is_stale:
            st.warning("⚠️ **Results are out of date.** Input settings or filter stack have changed. Click **🚀 Calculate Spectrum** to update.")
            
        cols_m = st.columns(4)
        cols_m[0].metric('Power at DUT', format_power(power_watts), delta=calc_data['delta_power_str'])
        cols_m[1].metric('Total Flux', f"{format_sci(total_flux)} ph/s", delta=calc_data['delta_flux_str'])

        if fluoro_df is not None:
            cols_m[2].metric('Fluorescence Power at DUT', format_power(f_power), delta=calc_data['f_delta_str'], delta_color='inverse')
            cols_m[3].metric('Fluorescence %', f"{pct_of_total:.3f}%")

        if scatter_df is not None:
            s_power = calc_data.get('s_power', 0.0)
            s_pct = calc_data.get('s_pct_of_total', 0.0)
            cols_s = st.columns(4)
            cols_s[0].metric('Total Secondary %', f"{(s_pct + pct_of_total):.3f}%",
                             help="Fluorescence + Compton scatter as a fraction of total power at the DUT. This is the 'fog' contribution from the filter stack acting as a secondary extended source.")
            cols_s[2].metric('Compton Scatter Power at DUT', format_power(s_power), delta=calc_data.get('s_delta_str'), delta_color='inverse')
            cols_s[3].metric('Scatter %', f"{s_pct:.3f}%")

        fig = go.Figure()
        input_x = input_df.iloc[:, 0].values
        input_y = input_df.iloc[:, 1].values

        in_x, in_y = display_xy(input_x, input_y)
        out_x, out_y = display_xy(res_df['Energy_KeV'].values, res_df['Photons_per_s'].values)

        fig.add_trace(go.Scatter(x=in_x, y=in_y, name='Input', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=out_x, y=out_y, name='Output', line=dict(color='blue', width=2)))

        max_y = np.nanmax(out_y) if len(out_y) else 1.0
        if pd.isna(max_y) or max_y <= 0:
            max_y = 1.0

        if fluoro_df is not None:
            f_x, f_y = display_xy(fluoro_df['Energy_KeV'].values, fluoro_df['Photons_per_s'].values)
            fig.add_trace(go.Scatter(x=f_x, y=f_y, name='Fluorescence', line=dict(color='red', width=2)))
            if len(f_y):
                max_y = max(max_y, np.nanmax(f_y))

        if scatter_df is not None and not scatter_df.empty:
            s_x, s_y = display_xy(scatter_df['Energy_KeV'].values, scatter_df['Photons_per_s'].values)
            fig.add_trace(go.Scatter(x=s_x, y=s_y, name='Compton Scatter', line=dict(color='orange', width=2)))

        x_max = float(np.nanmax(input_x)) * 1.02

        if log_y:
            log_top = np.log10(max(max_y, float(np.nanmax(in_y)), 1e-30))
            yaxis_cfg = dict(type='log', range=[log_top - 9, log_top + 0.3])
        else:
            actual_max = max(max_y, float(np.nanmax(in_y)))
            upper_bound = min(actual_max, max_y * 1.5)
            upper_bound = max(upper_bound, max_y)
            yaxis_cfg = dict(range=[0, upper_bound])

        fig.update_layout(
            xaxis_title='Energy (keV)',
            yaxis_title='Fluence (Photons/s/keV)',
            xaxis=dict(range=[0, x_max]),
            yaxis=yaxis_cfg,
            height=550,
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, width='stretch')

        dl_c1, dl_c2, dl_c3 = st.columns(3)
        dl_c1.download_button('Download Result CSV', res_df.to_csv(index=False), 'result.csv', 'text/csv')
        if fluoro_df is not None:
            dl_c2.download_button('Download Fluorescence CSV', fluoro_df.to_csv(index=False), 'result_fluoro.csv', 'text/csv')
        if scatter_df is not None and not scatter_df.empty:
            dl_c3.download_button('Download Scatter CSV', scatter_df.to_csv(index=False), 'result_scatter.csv', 'text/csv')
        
        # --- Fluorescence Optimizer ---
        with st.expander("✨ Cascading Fluorescence Optimizer", expanded=False):
            st.caption("Automatically find the best secondary cascading filter to eliminate fluorescence fog generated by your primary filter.")
            
            # Grab filter info from material 1 in the filter stack
            if len(st.session_state.filters) > 0:
                fluor_mat1 = st.session_state.filters[0]['material']
                fluor_thick1 = st.session_state.filters[0]['thickness']
                has_filter = True
            else:
                fluor_mat1 = None
                fluor_thick1 = None
                has_filter = False
                
            fluor_cols1 = st.columns([2, 1])
            if has_filter:
                fluor_cols1[0].markdown(f"**Primary Filter (from Stack):** `{fluor_mat1}` ({fluor_thick1:.2f} mm)")
                fluor_opt_btn = fluor_cols1[1].button('🚀 Run Fluorescence Optimizer', type='secondary', key='run_fluor_opt_btn', width='stretch')
            else:
                fluor_cols1[0].warning("⚠️ Filter stack is empty. Please add a filter to the stack first.")
                fluor_opt_btn = False
                
            if fluor_opt_btn and has_filter:
                if input_option == 'Generate with SpekPy':
                    try:
                        import spekpy
                        s = spekpy.Spek(kvp=spekpy_kvp, th=spekpy_th, targ=spekpy_targ, dk=spekpy_dk)
                        if tube_window == 'Beryllium (2mm)':
                            s.filter('Be', 2.0)
                        elif tube_window == 'Aluminum (1mm)':
                            s.filter('Al', 1.0)
                        energy, fluence = s.get_spectrum()

                        import ct_optimizer
                        df_fluor_res, top_fluor_res = ct_optimizer.run_fluorescence_optimization(
                            energies=energy,
                            raw_photons=fluence,
                            mat1=fluor_mat1,
                            thick1=fluor_thick1,
                            candidate_materials=list(st.session_state.enabled_materials)
                        )
                        
                        st.session_state.fluor_opt_results = {
                            'df_fluor_res': df_fluor_res,
                            'top_fluor_res': top_fluor_res,
                            'mat1': fluor_mat1,
                            'thick1': fluor_thick1,
                            'candidates': list(st.session_state.enabled_materials),
                            'input_option': input_option,
                            'spekpy_kvp': spekpy_kvp,
                            'spekpy_th': spekpy_th,
                            'spekpy_targ': spekpy_targ,
                            'spekpy_dk': spekpy_dk,
                            'tube_window': tube_window
                        }
                    except Exception as e:
                        st.error(f"Fluorescence optimization failed: {e}")
                else:
                    st.error("The automated fluorescence optimizer currently requires the 'Generate with SpekPy' input option.")
                    
            if st.session_state.fluor_opt_results is not None:
                res = st.session_state.fluor_opt_results
                
                # Check if optimizer results are stale
                fluor_stale = False
                if (res['mat1'] != fluor_mat1 or
                    res['thick1'] != fluor_thick1 or
                    res.get('candidates') != st.session_state.enabled_materials or
                    res['input_option'] != input_option or
                    res['spekpy_kvp'] != spekpy_kvp or
                    res['spekpy_th'] != spekpy_th or
                    res['spekpy_targ'] != spekpy_targ or
                    res.get('spekpy_dk') != spekpy_dk or
                    res['tube_window'] != tube_window):
                    fluor_stale = True
                    
                if fluor_stale:
                    st.warning("⚠️ **Optimization results are out of date.** Settings or Primary Filter in Stack have changed. Click 'Run Fluorescence Optimizer' to update.")
                    
                df_fluor_res = res['df_fluor_res']
                top_fluor_res = res['top_fluor_res']
                
                if df_fluor_res.empty:
                    st.warning("The primary filter absorbs all energy or produces no fluorescence. No secondary filter needed.")
                else:
                    st.write(f"Top secondary filters to suppress {res['mat1']} fluorescence while preserving main beam power:")
                    
                    st.dataframe(top_fluor_res[['Filter_Material', 'Thickness_mm', 'F_elim', 'P_kept', 'Score']].style.format({
                        'Thickness_mm': '{:.2f}',
                        'F_elim': '{:.1%}',
                        'P_kept': '{:.1%}',
                        'Score': '{:.4f}'
                    }))

                    st.markdown("### 📥 Apply Secondary Filter")
                    st.write(f"Insert a secondary filter into the stack directly behind your `{res['mat1']}` primary. Clicking another option swaps it out, so you can quickly compare candidates:")
                    cols_apply_fl = st.columns(4)
                    for idx, (_, row) in enumerate(top_fluor_res.iterrows()):
                        mat = row['Filter_Material']
                        thick = row['Thickness_mm']
                        col_idx = idx % 4
                        if cols_apply_fl[col_idx].button(f"Apply {mat} ({thick:.2f} mm)", key=f"apply_fluor_{mat}", width='stretch'):
                            new_secondary = {'material': mat, 'thickness': round(float(thick), 3)}
                            prev_applied = st.session_state.get('fluor_applied_secondary')
                            if (prev_applied is not None and
                                len(st.session_state.filters) > 1 and
                                st.session_state.filters[1] == prev_applied):
                                st.session_state.filters[1] = new_secondary
                            else:
                                st.session_state.filters.insert(1, new_secondary)
                            st.session_state.fluor_applied_secondary = new_secondary
                            sync_filter_widget_state()
                            save_filters()
                            st.session_state.calculated_spectrum = None
                            st.rerun()

                    st.write("### Trade-off Analysis (Pareto Front)")
                    st.caption("Hover over points to see exact material and thickness. Top left is best.")
                    
                    fig_fluor = go.Figure()
                    for m in df_fluor_res['Filter_Material'].unique():
                        m_df = df_fluor_res[df_fluor_res['Filter_Material'] == m]
                        fig_fluor.add_trace(go.Scatter(
                            x=1.0 - m_df['P_kept'], 
                            y=m_df['F_elim'], 
                            mode='lines+markers',
                            name=m,
                            text=[f"{m} {t:.2f}mm" for t in m_df['Thickness_mm']],
                            hovertemplate="%{text}<br>Beam Removed: %{x:.1%}<br>Fluor Elim: %{y:.1%}<extra></extra>"
                        ))
                        
                    fig_fluor.update_layout(
                        title="Fluorescence Eliminated vs Main Beam Removed",
                        xaxis_title="% Main Beam Removed",
                        yaxis_title="% Fluorescence Eliminated",
                        xaxis=dict(tickformat=".0%"),
                        yaxis=dict(tickformat=".0%"),
                        legend_title="Material"
                    )
                    st.plotly_chart(fig_fluor, width='stretch')
                    
        st.markdown('---')
        st.subheader('Target Contrast & Optimization')
        tab_contrast, tab_opt, tab_emission = st.tabs(['Contrast Analysis', 'Filter Optimizer', 'Emission Line Optimizer'])
        
        with tab_contrast:
            st.caption("The Contrast Analysis runs automatically when you click **Calculate Spectrum**. It compares the energy deposited by the beam passing through Target A vs Target B based on your current Filter Stack.")
            try:
                import ct_physics
                contrast_res = ct_physics.calculate_contrast(
                    spectrum_df=res_df,
                    target_a_mat=target_a_mat,
                    target_a_thick=target_a_thick,
                    target_b_mat=target_b_mat,
                    target_b_thick=target_b_thick,
                    scint_mat=scint_mat,
                    scint_thick=scint_thick
                )
                
                ccols = st.columns(3)
                ccols[0].metric(f'Signal through A ({target_a_mat})', f"{format_sci(contrast_res['signal_A'])} keV/s")
                ccols[1].metric(f'Signal through B ({target_b_mat})', f"{format_sci(contrast_res['signal_B'])} keV/s")
                ccols[2].metric('Weber Contrast |A-B|/A', f"{contrast_res['contrast']*100:.2f} %")
                
                fig_ct = go.Figure()
                df_A = contrast_res['df_A']
                df_B = contrast_res['df_B']

                ct_ax, ct_ay = display_xy(df_A['Energy_KeV'].values, df_A['Energy_Deposited_keV_per_s'].values)
                ct_bx, ct_by = display_xy(df_B['Energy_KeV'].values, df_B['Energy_Deposited_keV_per_s'].values)

                fig_ct.add_trace(go.Scatter(
                    x=ct_ax, y=ct_ay,
                    name=f'{target_a_mat} ({target_a_thick}mm)', line=dict(color='orange', width=2)
                ))
                fig_ct.add_trace(go.Scatter(
                    x=ct_bx, y=ct_by,
                    name=f'{target_b_mat} ({target_b_thick}mm)', line=dict(color='teal', width=2)
                ))
                fig_ct.add_trace(go.Scatter(
                    x=ct_ax, y=np.abs(ct_ay - ct_by),
                    name='|A − B| (Contrast driver)', line=dict(color='red', width=1.5, dash='dash')
                ))

                fig_ct.update_layout(
                    title='Energy Deposited in Scintillator (Signal)',
                    xaxis_title='Energy (keV)',
                    yaxis_title='Deposited Energy (keV/s)',
                    xaxis=dict(range=[0, float(np.nanmax(df_A['Energy_KeV'].values)) * 1.02]),
                    legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_ct, width='stretch')
            except Exception as e:
                st.error(f"Contrast calculation failed: {e}")
                
        with tab_opt:
            st.caption("Automatically find the best filter to penetrate shadowing material while maximizing Contrast-to-Noise Ratio (CNR).")
            
            opt_cols1 = st.columns(4)
            opt_shadow1_mat = opt_cols1[0].selectbox('Substrate Material', options=ct_materials.CT_MATERIALS, index=0, key='opt_shadow1_mat')
            opt_shadow1_thick = opt_cols1[1].number_input('Substrate Thick (mm)', value=1.6, min_value=0.0, format='%.2f', step=0.1, key='opt_shadow1_thick')
            opt_shadow2_mat = opt_cols1[2].selectbox('Obstruction Material', options=ct_materials.CT_MATERIALS, index=7, key='opt_shadow2_mat')
            opt_shadow2_thick = opt_cols1[3].number_input('Obstruction Thick (mm)', value=0.5, min_value=0.0, format='%.2f', step=0.1, key='opt_shadow2_thick')
            
            opt_cols2 = st.columns(4)
            opt_path_mult = opt_cols2[0].number_input('Aspect Angle Path Multiplier', value=3.0, min_value=1.0, step=0.5, help="Simulates passing through the shadow at a steep angle. E.g. 3.0 represents ~70 degrees, aligning multiple pins.", key='opt_path_mult')
            opt_time_limit = opt_cols2[1].number_input('Max Time Penalty (x)', value=5.0, min_value=1.0, max_value=100.0, step=0.5, help="The maximum acceptable exposure time penalty. Filters exceeding this are rejected.", key='opt_time_limit')
            opt_custom_filters_str = opt_cols2[2].text_input('Custom Filters', value='', help='Space separated elements or Z#s (e.g. 29 Au C)', key='opt_custom_filters')
            
            opt_cols2[3].markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
            optimize_btn = opt_cols2[3].button('🚀 Run Filter Optimizer', type='secondary', key='run_filter_opt_btn', width='stretch')
            
            if optimize_btn:
                st.info("Running Penetration vs Contrast Filter Optimizer...")
                if input_option == 'Generate with SpekPy':
                    try:
                        import spekpy
                        s = spekpy.Spek(kvp=spekpy_kvp, th=spekpy_th, targ=spekpy_targ, dk=spekpy_dk)
                        if tube_window == 'Beryllium (2mm)':
                            s.filter('Be', 2.0)
                        elif tube_window == 'Aluminum (1mm)':
                            s.filter('Al', 1.0)
                        energy, fluence = s.get_spectrum()

                        import ct_optimizer
                        import importlib
                        importlib.reload(ct_optimizer)
                        import apply_filter_and_fluorescence
                        opt_custom_filters = []
                        if opt_custom_filters_str.strip():
                            for item in opt_custom_filters_str.split():
                                _, sym = apply_filter_and_fluorescence.get_element_properties(item)
                                if sym and sym != "Unknown":
                                    opt_custom_filters.append(sym)
                                    
                        df_opt_res, top_opt_res = ct_optimizer.run_filter_optimization(
                            energies=energy,
                            raw_photons=fluence,
                            target_a_mat=target_a_mat,
                            target_a_thick=target_a_thick,
                            target_b_mat=target_b_mat,
                            target_b_thick=target_b_thick,
                            shadow1_mat=opt_shadow1_mat,
                            shadow1_thick=opt_shadow1_thick,
                            shadow2_mat=opt_shadow2_mat,
                            shadow2_thick=opt_shadow2_thick,
                            scint_mat=scint_mat,
                            scint_thick=scint_thick,
                            path_multiplier=opt_path_mult,
                            time_penalty_limit=opt_time_limit,
                            extra_filters=opt_custom_filters,
                            filter_materials=list(st.session_state.enabled_materials)
                        )
                        
                        z_map = apply_filter_and_fluorescence.SYM_TO_Z
                        if 'Z' not in top_opt_res.columns:
                            top_opt_res['Z'] = top_opt_res['Filter_Material'].map(z_map).fillna(0).astype(int)
                            top_opt_res = top_opt_res.set_index('Z')
                        if 'Plot' not in top_opt_res.columns:
                            top_opt_res.insert(0, 'Plot', [True if i < 3 else False for i in range(len(top_opt_res))])
                        
                        st.session_state.filter_opt_results = {
                            'df_opt_res': df_opt_res,
                            'top_opt_res': top_opt_res,
                            'candidates': list(st.session_state.enabled_materials),
                            'shadow1_mat': opt_shadow1_mat,
                            'shadow1_thick': opt_shadow1_thick,
                            'shadow2_mat': opt_shadow2_mat,
                            'shadow2_thick': opt_shadow2_thick,
                            'path_multiplier': opt_path_mult,
                            'time_penalty_limit': opt_time_limit,
                            'target_a_mat': target_a_mat,
                            'target_a_thick': target_a_thick,
                            'target_b_mat': target_b_mat,
                            'target_b_thick': target_b_thick,
                            'scint_mat': scint_mat,
                            'scint_thick': scint_thick,
                            'input_option': input_option,
                            'spekpy_kvp': spekpy_kvp,
                            'spekpy_th': spekpy_th,
                            'spekpy_targ': spekpy_targ,
                            'spekpy_dk': spekpy_dk,
                            'tube_window': tube_window
                        }
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                else:
                    st.error("The automated filter optimizer currently requires the 'Generate with SpekPy' input option.")
                    
            if st.session_state.filter_opt_results is not None:
                res = st.session_state.filter_opt_results
                
                # Check if filter optimizer results are stale
                filter_opt_stale = False
                if (res.get('candidates') != st.session_state.enabled_materials or
                    res.get('shadow1_mat') != opt_shadow1_mat or
                    res.get('shadow1_thick') != opt_shadow1_thick or
                    res.get('shadow2_mat') != opt_shadow2_mat or
                    res.get('shadow2_thick') != opt_shadow2_thick or
                    res.get('time_penalty_limit') != opt_time_limit or
                    res['path_multiplier'] != opt_path_mult or
                    res['target_a_mat'] != target_a_mat or
                    res['target_a_thick'] != target_a_thick or
                    res['target_b_mat'] != target_b_mat or
                    res['target_b_thick'] != target_b_thick or
                    res['scint_mat'] != scint_mat or
                    res['scint_thick'] != scint_thick or
                    res['input_option'] != input_option or
                    res['spekpy_kvp'] != spekpy_kvp or
                    res['spekpy_th'] != spekpy_th or
                    res['spekpy_targ'] != spekpy_targ or
                    res.get('spekpy_dk') != spekpy_dk or
                    res['tube_window'] != tube_window):
                    filter_opt_stale = True
                    
                if filter_opt_stale:
                    st.warning("⚠️ **Optimization results are out of date.** Settings have changed. Click 'Run Filter Optimizer' to update.")
                    
                df_opt_res = res['df_opt_res']
                top_opt_res = res['top_opt_res']
                
                st.subheader('Optimization Results')
                st.write(f"Top filter thicknesses balancing Dynamic Contrast against detector saturation limits (Max {opt_time_limit}x Time Penalty):")
                
                edited_top_opt_res = st.data_editor(
                    top_opt_res[['Plot', 'Filter_Material', 'Thickness_mm', 'Dynamic_CNR', 'Time_Penalty']],
                    column_config={
                        'Plot': st.column_config.CheckboxColumn('Plot', help="Select to display on the chart below"),
                        'Filter_Material': 'Material',
                        'Thickness_mm': st.column_config.NumberColumn('Thickness (mm)', format='%.2f'),
                        'Dynamic_CNR': st.column_config.NumberColumn('Dynamic CNR', format='%.8f'),
                        'Time_Penalty': st.column_config.NumberColumn('Time Penalty', format='%.1fx')
                    },
                    disabled=['Filter_Material', 'Thickness_mm', 'Dynamic_CNR', 'Time_Penalty'],
                    hide_index=False,
                    width='stretch',
                    key='opt_data_editor'
                )
                st.markdown("### 📥 Apply Optimized Filter")
                st.write("Set an optimized filter as your active filter stack (automatically enables fluorescence analysis):")
                cols_apply = st.columns(4)
                for idx, (_, row) in enumerate(edited_top_opt_res.iterrows()):
                    mat = row['Filter_Material']
                    thick = row['Thickness_mm']
                    col_idx = idx % 4
                    if cols_apply[col_idx].button(f"Apply {mat} ({thick:.2f} mm)", key=f"apply_opt_{mat}", width='stretch'):
                        st.session_state.filters = [{'material': mat, 'thickness': round(float(thick), 3)}]
                        sync_filter_widget_state()
                        save_filters()
                        st.session_state.calculated_spectrum = None
                        st.rerun()
                
                top_materials = edited_top_opt_res[edited_top_opt_res['Plot']]['Filter_Material'].values
                from plotly.subplots import make_subplots
                fig_opt = make_subplots(specs=[[{"secondary_y": True}]])
                
                for m in top_materials:
                    m_df = df_opt_res[(df_opt_res['Filter_Material'] == m) & (df_opt_res['Time_Penalty'] <= 12.0)]
                    fig_opt.add_trace(go.Scatter(x=m_df['Time_Penalty'], y=m_df['Dynamic_CNR'], name=f'{m} Dynamic CNR'), secondary_y=False)
                    fig_opt.add_trace(go.Scatter(x=m_df['Time_Penalty'], y=m_df['Thickness_mm'], name=f'{m} Thickness', line=dict(dash='dot'), visible='legendonly'), secondary_y=True)
                    
                fig_opt.update_layout(
                    title='Time Penalty vs Dynamic-Range-Limited CNR',
                    legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig_opt.update_yaxes(title_text="Contrast-to-Noise Ratio (CNR)", secondary_y=False)
                fig_opt.update_yaxes(title_text="Filter Thickness (mm)", secondary_y=True)
                fig_opt.update_xaxes(title_text="Time Penalty (x)", range=[1.0, 5.0])
                
                st.plotly_chart(fig_opt, width='stretch')
        with tab_emission:
            st.markdown("### Emission Line Optimizer")
            st.write("This optimizer evaluates all practical filter materials (elements available as thin foils or discs) across a range of thicknesses to find the best filter for stripping out low-energy background radiation while preserving the Tungsten (W) K-emission lines ($\ge$ 57 keV).")
            
            e_col1, e_col2 = st.columns(2)
            opt_kvp = e_col1.number_input("Tube Voltage (kVp)", min_value=60, max_value=450, value=220, step=10, key='opt_em_kvp')
            opt_max_t = e_col2.number_input("Max Thickness (mm)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key='opt_em_thick')
            
            if st.button("Run Emission Optimizer", type="primary"):
                with st.spinner("Sweeping practical filter materials..."):
                    try:
                        from emission_optimizer import run_emission_optimizer
                        df_em = run_emission_optimizer(kvp=opt_kvp, max_thickness_mm=opt_max_t)
                        st.session_state.emission_opt_results = df_em
                        st.session_state.emission_opt_kvp = opt_kvp
                        st.session_state.emission_opt_max_t = opt_max_t
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
            
            if 'emission_opt_results' in st.session_state and st.session_state.emission_opt_results is not None:
                if st.session_state.emission_opt_kvp != opt_kvp or st.session_state.emission_opt_max_t != opt_max_t:
                    st.warning("⚠️ **Settings have changed.** Click 'Run Emission Optimizer' to update results.")
                    
                df_em = st.session_state.emission_opt_results
                
                st.write("Best thickness for each material, ranked by Score = (Removed Low %) * (Kept High %)^2")
                edited_df_em = st.data_editor(
                    df_em.head(50),
                    column_config={
                        'Filter_Material': 'Material',
                        'Thickness_mm': st.column_config.NumberColumn('Thickness (mm)', format='%.2f'),
                        'Removed_Low_Pct': st.column_config.NumberColumn('Removed <57 keV (%)', format='%.2f %%'),
                        'Kept_High_Pct': st.column_config.NumberColumn('Kept >=57 keV (%)', format='%.2f %%'),
                        'Score': st.column_config.NumberColumn('Score', format='%.0f')
                    },
                    disabled=True,
                    hide_index=True,
                    width='stretch',
                    key='em_data_editor'
                )
                
                st.markdown("### 📥 Apply Optimized Filter")
                cols_apply_em = st.columns(4)
                for idx, (_, row) in enumerate(edited_df_em.head(8).iterrows()):
                    mat = row['Filter_Material']
                    thick = row['Thickness_mm']
                    col_idx = idx % 4
                    if cols_apply_em[col_idx].button(f"Apply {mat} ({thick:.2f} mm)", key=f"apply_em_{mat}_{idx}", width='stretch'):
                        st.session_state.filters = [{'material': mat, 'thickness': round(float(thick), 3)}]
                        sync_filter_widget_state()
                        save_filters()
                        st.session_state.calculated_spectrum = None
                        st.rerun()

    else:
        st.info('Configure filters and click "Calculate Spectrum" to see the result.')
