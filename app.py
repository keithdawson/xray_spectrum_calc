import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import io
import json
from apply_filter_and_fluorescence import (
    run_stack,
    PRACTICAL_FILTER_MATERIALS,
    SYM_TO_Z,
    acceptance_cone_from_geometry,
)
from bundled_data import DEFAULT_SPECTRA
from source_spectrum import generate_spekpy_spectrum, spectrum_normalization
from ct_physics import attenuation_exponent
from spectrum_utils import energy_bin_widths

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
        bb_head1, bb_head2 = st.columns([1.4, 1.0])
        bb_head1.markdown("**Beam Basics**")
        with bb_head2.popover('📐 Cone Geometry'):
            st.caption('Compute the secondary acceptance cone full angle from physical Filter-to-Detector distance $L$ and detector aperture diameter $D$ via $\\theta = 2\\arctan(D / 2L)$.')
            use_phys_cone = st.checkbox(
                'Derive Cone from Detector Geometry',
                value=False,
                key='use_phys_cone',
            )
            filter_to_det_mm = st.number_input(
                'Filter-to-Detector Distance (mm)',
                min_value=1.0,
                value=500.0,
                step=10.0,
                format='%.1f',
                key='filter_to_det_mm',
                help='Distance from the filter exit plane to the detector entrance plane.',
            )
            det_aperture_mm = st.number_input(
                'Detector Aperture Diameter (mm)',
                min_value=0.1,
                value=200.0,
                step=10.0,
                format='%.1f',
                key='det_aperture_mm',
                help='Effective active diameter or width of the detector / collimator aperture.',
            )
            derived_cone_deg = acceptance_cone_from_geometry(filter_to_det_mm, det_aperture_mm)
            st.caption(f'Derived full acceptance cone: **{derived_cone_deg:.2f}°**')
        bb_c1, bb_c2 = st.columns(2)
        with bb_c1:
            if use_phys_cone:
                beam_angle = float(derived_cone_deg)
                st.number_input(
                    'Secondary Acceptance Cone (deg)',
                    value=round(beam_angle, 2),
                    disabled=True,
                    help='Computed automatically from Filter-to-Detector Distance and Detector Aperture Diameter in 📐 Cone Geometry.',
                )
            else:
                beam_angle = st.slider('Secondary Acceptance Cone (deg)', 1.0, 180.0, 30.0, step=1.0,
                                       help='Full acceptance angle about each incident ray, or use 📐 Cone Geometry to compute from physical filter-to-detector distance and aperture size.')
            tube_current_ma = st.number_input('Tube Current (mA)', min_value=0.001, value=1.0, step=0.1, format='%.3f',
                                              help='Used for SpekPy spectra. Generated output is photons per cm² per second.')
        with bb_c2:
            tube_window = st.selectbox('Tube Window', options=['Beryllium (2mm)', 'Aluminum (1mm)', 'None'], index=1)
            distance_to_target = st.number_input('Distance to Target (mm)', value=1000.0, min_value=1.0, step=10.0, format='%.1f',
                                                  help='SpekPy uses this source distance for inverse-square fluence. The air path is included for all inputs; imported spectra retain their supplied normalization.')
        
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
        if len(E) < 3 or np.median(energy_bin_widths(E)) >= bin_kev:
            return E, phi
        edges = np.arange(E.min() - 1e-9, E.max() + bin_kev, bin_kev)
        integral, _ = np.histogram(E, bins=edges, weights=phi * energy_bin_widths(E))
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
        'tube_current_ma': tube_current_ma,
        'source_distance_mm': distance_to_target,
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
            energy, fluence = generate_spekpy_spectrum(
                spekpy_kvp, spekpy_th, spekpy_targ, spekpy_dk,
                tube_current_ma, distance_to_target, tube_window)
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
                    normalization = spectrum_normalization(input_df, input_option == 'Generate with SpekPy')
                    if st.session_state.get('previous_normalization') != normalization:
                        for key in ('prev_power_watts', 'prev_total_flux', 'prev_f_power', 'prev_s_power'):
                            st.session_state.pop(key, None)
                    st.session_state.previous_normalization = normalization
                    
                    # Calculate Power at Target
                    E_out = res_df['Energy_KeV'].values
                    Phi_out = res_df['Photons_per_s'].values
                    dE_out = energy_bin_widths(E_out)
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
                                f_dE = energy_bin_widths(f_E)
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
                                s_dE = energy_bin_widths(s_E)
                                s_power = np.sum(s_Phi * s_dE * s_E * 1.60218e-16)
                                s_flux = np.sum(s_Phi * s_dE)
                                s_pct_of_total = (s_power / power_watts) * 100.0 if power_watts > 0 else 0.0

                                if 'prev_s_power' in st.session_state and st.session_state.prev_s_power > 0:
                                    pct_s = (s_power - st.session_state.prev_s_power) / st.session_state.prev_s_power * 100.0
                                    s_delta_str = f"{pct_s:.2f}%"
                                st.session_state.prev_s_power = s_power

                    for frame in (res_df, fluoro_df, scatter_df):
                        if frame is not None:
                            frame['Normalization'] = normalization
                    st.session_state.calculated_spectrum = {
                        'normalization': normalization,
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
                p1['tube_current_ma'] != p2.get('tube_current_ma') or
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
            
        normalization = calc_data.get('normalization', 'photons/s/keV')
        per_area = normalization == 'photons/cm2/s/keV'
        relative = normalization == 'relative/keV'
        flux_unit = 'a.u.' if relative else ('ph/cm²/s' if per_area else 'ph/s')
        signal_unit = 'keV·a.u.' if relative else ('keV/cm²/s' if per_area else 'keV/s')
        def display_power(value):
            if relative:
                return f'{format_sci(value / 1.60218e-16)} keV·a.u.'
            return format_power(value) + ('/cm²' if per_area else '')
        st.caption('Generated spectra use tube current and inverse-square distance scaling. CSV downloads preserve normalization metadata. Imported spectra without metadata retain the units implied by their input column.')
        st.caption('Secondary radiation is a plane-slab cone estimate including K-shell fluorescence, incoherent (Compton) scatter, and coherent (Rayleigh) scatter. Object scatter, L-shell emission, and multiple interactions within each foil are not modeled.')
        cols_m = st.columns(4)
        cols_m[0].metric('Energy output at DUT' if relative else ('Power density at DUT' if per_area else 'Power at DUT'), display_power(power_watts), delta=calc_data['delta_power_str'])
        cols_m[1].metric('Photon fluence rate' if per_area else 'Integrated photon output', f"{format_sci(total_flux)} {flux_unit}", delta=calc_data['delta_flux_str'])

        if fluoro_df is not None:
            cols_m[2].metric('Fluorescence energy output', display_power(f_power), delta=calc_data['f_delta_str'], delta_color='inverse')
            cols_m[3].metric('Fluorescence %', f"{pct_of_total:.3f}%")

        if scatter_df is not None:
            s_power = calc_data.get('s_power', 0.0)
            s_pct = calc_data.get('s_pct_of_total', 0.0)
            cols_s = st.columns(4)
            cols_s[0].metric('Total Secondary %', f"{(s_pct + pct_of_total):.3f}%",
                             help="Accepted K fluorescence + Compton/Rayleigh scatter as a fraction of modeled total energy fluence. This is a cone estimate, not a spatial blur or detector fog measurement.")
            cols_s[2].metric('Scatter energy output (Compton + Rayleigh)', display_power(s_power), delta=calc_data.get('s_delta_str'), delta_color='inverse')
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
            fig.add_trace(go.Scatter(x=s_x, y=s_y, name='Scatter (Compton + Rayleigh)', line=dict(color='orange', width=2)))

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
            yaxis_title=f'Spectral density ({flux_unit}/keV)',
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
        
        def load_active_input_spectrum():
            if input_option == 'Generate with SpekPy':
                return generate_spekpy_spectrum(
                    spekpy_kvp, spekpy_th, spekpy_targ, spekpy_dk,
                    tube_current_ma, distance_to_target, tube_window)
            if embedded_content:
                df_in = pd.read_csv(io.StringIO(embedded_content))
            elif spectrum_path:
                df_in = pd.read_csv(spectrum_path)
            elif uploaded_file is not None:
                df_in = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
            else:
                df_in = input_df.copy()
            if 'Energy_KeV' not in df_in.columns and 'Energy_keV' in df_in.columns:
                df_in = df_in.rename(columns={'Energy_keV': 'Energy_KeV'})
            if 'Photons_per_s' not in df_in.columns and 'Relative_Intensity' in df_in.columns:
                df_in = df_in.rename(columns={'Relative_Intensity': 'Photons_per_s'})
            return df_in['Energy_KeV'].values.astype(float), df_in['Photons_per_s'].values.astype(float)

        # --- Fluorescence Optimizer ---
        with st.expander("✨ Cascading Fluorescence Optimizer", expanded=False):
            st.caption("Automatically find the best downstream cascading filter to eliminate fluorescence (and optionally Compton scatter) fog generated by your upstream filter stack.")
            
            has_filter = len(st.session_state.filters) > 0
            fluor_cols1 = st.columns([1.6, 1.1, 1.0])
            fluor_include_scatter = fluor_cols1[1].checkbox(
                'Include Compton Scatter Fog',
                value=False,
                key='fluor_include_scatter',
                help='When enabled, ranks secondary filters by total secondary energy removed (K-fluorescence + Compton scatter) rather than K-fluorescence alone, preventing thick low-Z filters with heavy Compton scatter from ranking artificially high.'
            )
            if has_filter:
                def _format_upstream_depth(depth):
                    chain = " → ".join(
                        f"{f['material']} ({f['thickness']:.2f} mm)"
                        for f in st.session_state.filters[:depth]
                    )
                    label = "Layer 1" if depth == 1 else f"Layers 1–{depth}"
                    return f"{label}: {chain}"

                if len(st.session_state.filters) == 1:
                    fluor_upstream_depth = 1
                    fluor_cols1[0].markdown(f"**Upstream Stack:** `{_format_upstream_depth(1)}`")
                else:
                    fluor_upstream_depth = fluor_cols1[0].selectbox(
                        'Optimize Behind Upstream Stack',
                        options=list(range(1, len(st.session_state.filters) + 1)),
                        index=0,
                        format_func=_format_upstream_depth,
                        key='fluor_upstream_depth_sel',
                        help='Choose Layer 1 to optimize the second filter, or Layers 1–N to cascade a third/fourth graded-Z filter behind the existing stack.'
                    )
                selected_upstream = [
                    {'material': f['material'], 'thickness': float(f['thickness'])}
                    for f in st.session_state.filters[:fluor_upstream_depth]
                ]
                fluor_stack_summary = " → ".join(
                    f"{f['material']} ({f['thickness']:.2f} mm)" for f in selected_upstream
                )
                fluor_opt_btn = fluor_cols1[2].button('🚀 Run Fluorescence Optimizer', type='secondary', key='run_fluor_opt_btn', width='stretch')
            else:
                selected_upstream = []
                fluor_stack_summary = ""
                fluor_cols1[0].warning("⚠️ Filter stack is empty. Please add a filter to the stack first.")
                fluor_opt_btn = False
                
            if fluor_opt_btn and has_filter:
                try:
                    energy, fluence = load_active_input_spectrum()

                    import ct_optimizer
                    df_fluor_res, top_fluor_res = ct_optimizer.run_fluorescence_optimization(
                        energies=energy,
                        raw_photons=fluence,
                        mat1=selected_upstream[0]['material'],
                        thick1=selected_upstream[0]['thickness'],
                        candidate_materials=list(st.session_state.enabled_materials),
                        geom_params={'beam_angle_deg': beam_angle},
                        downstream_transmission=np.exp(-attenuation_exponent('Air', energy, distance_to_target)),
                        include_scatter=fluor_include_scatter,
                        upstream_filters=selected_upstream,
                    )
                    
                    st.session_state.fluor_opt_results = {
                        'df_fluor_res': df_fluor_res,
                        'top_fluor_res': top_fluor_res,
                        'mat1': selected_upstream[-1]['material'],
                        'thick1': selected_upstream[-1]['thickness'],
                        'upstream_filters': selected_upstream,
                        'stack_summary': fluor_stack_summary,
                        'insert_idx': len(selected_upstream),
                        'beam_angle': beam_angle,
                        'include_scatter': fluor_include_scatter,
                        'candidates': list(st.session_state.enabled_materials),
                        'input_option': input_option,
                        'uploaded_file_name': uploaded_file.name if uploaded_file else None,
                        'spekpy_kvp': spekpy_kvp,
                        'spekpy_th': spekpy_th,
                        'spekpy_targ': spekpy_targ,
                        'spekpy_dk': spekpy_dk,
                        'tube_window': tube_window, 'tube_current_ma': tube_current_ma,
                        'source_distance_mm': distance_to_target
                    }
                    st.session_state.fluor_applied_secondary = None
                except Exception as e:
                    st.error(f"Fluorescence optimization failed: {e}")
                    
            if st.session_state.fluor_opt_results is not None:
                res = st.session_state.fluor_opt_results
                
                # Check if optimizer results are stale
                fluor_stale = False
                if (res.get('upstream_filters') != selected_upstream or
                    res.get('beam_angle') != beam_angle or
                    res.get('include_scatter', False) != fluor_include_scatter or
                    res.get('candidates') != st.session_state.enabled_materials or
                    res['input_option'] != input_option or
                    res.get('uploaded_file_name') != (uploaded_file.name if uploaded_file else None) or
                    res['spekpy_kvp'] != spekpy_kvp or
                    res['spekpy_th'] != spekpy_th or
                    res['spekpy_targ'] != spekpy_targ or
                    res.get('spekpy_dk') != spekpy_dk or
                    res['tube_window'] != tube_window or
                    res.get('tube_current_ma') != tube_current_ma or
                    res.get('source_distance_mm') != distance_to_target):
                    fluor_stale = True
                    
                if fluor_stale:
                    st.warning("⚠️ **Optimization results are out of date.** Settings or Upstream Filter Stack have changed. Click 'Run Fluorescence Optimizer' to update.")
                    
                df_fluor_res = res['df_fluor_res']
                top_fluor_res = res['top_fluor_res']
                used_scatter = res.get('include_scatter', False)
                summary_label = res.get('stack_summary', res['mat1'])
                
                if df_fluor_res.empty:
                    st.warning("The upstream filter stack absorbs all energy or produces no secondary radiation. No downstream filter needed.")
                else:
                    target_desc = "total secondary (fluorescence + scatter)" if used_scatter else "fluorescence"
                    st.write(f"Top cascading filters to suppress `{summary_label}` {target_desc} while preserving main beam power:")
                    
                    if used_scatter and 'Secondary_elim' in top_fluor_res.columns:
                        st.dataframe(top_fluor_res[['Filter_Material', 'Thickness_mm', 'F_elim', 'Secondary_elim', 'P_kept', 'Score']].style.format({
                            'Thickness_mm': '{:.2f}',
                            'F_elim': '{:.1%}',
                            'Secondary_elim': '{:.1%}',
                            'P_kept': '{:.1%}',
                            'Score': '{:.4f}'
                        }))
                    else:
                        st.dataframe(top_fluor_res[['Filter_Material', 'Thickness_mm', 'F_elim', 'P_kept', 'Score']].style.format({
                            'Thickness_mm': '{:.2f}',
                            'F_elim': '{:.1%}',
                            'P_kept': '{:.1%}',
                            'Score': '{:.4f}'
                        }))

                    insert_idx = int(res.get('insert_idx', 1))
                    st.markdown("### 📥 Apply Cascading Filter")
                    st.write(f"Insert a cascading filter at **Layer {insert_idx + 1}** directly behind `{summary_label}`. Clicking another option swaps it out so you can quickly compare candidates:")
                    cols_apply_fl = st.columns(4)
                    for idx, (_, row) in enumerate(top_fluor_res.iterrows()):
                        mat = row['Filter_Material']
                        thick = row['Thickness_mm']
                        col_idx = idx % 4
                        if cols_apply_fl[col_idx].button(f"Apply {mat} ({thick:.2f} mm)", key=f"apply_fluor_{mat}", width='stretch'):
                            new_secondary = {'material': mat, 'thickness': round(float(thick), 3)}
                            prev_applied = st.session_state.get('fluor_applied_secondary')
                            if (prev_applied is not None and
                                len(st.session_state.filters) > insert_idx and
                                st.session_state.filters[insert_idx] == prev_applied):
                                st.session_state.filters[insert_idx] = new_secondary
                            else:
                                st.session_state.filters.insert(min(insert_idx, len(st.session_state.filters)), new_secondary)
                            st.session_state.fluor_applied_secondary = new_secondary
                            sync_filter_widget_state()
                            save_filters()
                            st.session_state.calculated_spectrum = None
                            st.rerun()

                    st.write("### Trade-off Analysis (Pareto Front)")
                    st.caption("Hover over points to see exact material and thickness. Top left is best.")
                    
                    fig_fluor = go.Figure()
                    elim_col = 'Secondary_elim' if (used_scatter and 'Secondary_elim' in df_fluor_res.columns) else 'F_elim'
                    elim_label = 'Total Secondary Eliminated' if elim_col == 'Secondary_elim' else 'Fluorescence Eliminated'
                    for m in df_fluor_res['Filter_Material'].unique():
                        m_df = df_fluor_res[df_fluor_res['Filter_Material'] == m]
                        fig_fluor.add_trace(go.Scatter(
                            x=1.0 - m_df['P_kept'], 
                            y=m_df[elim_col], 
                            mode='lines+markers',
                            name=m,
                            text=[f"{m} {t:.2f}mm" for t in m_df['Thickness_mm']],
                            hovertemplate="%{text}<br>Beam Removed: %{x:.1%}<br>" + elim_label + ": %{y:.1%}<extra></extra>"
                        ))
                        
                    fig_fluor.update_layout(
                        title=f"{elim_label} vs Main Beam Removed",
                        xaxis_title="% Main Beam Removed",
                        yaxis_title=f"% {elim_label}",
                        xaxis=dict(tickformat=".0%"),
                        yaxis=dict(tickformat=".0%"),
                        legend_title="Material"
                    )
                    st.plotly_chart(fig_fluor, width='stretch')
                    
        st.markdown('---')
        st.subheader('Target Contrast & Optimization')
        tab_contrast, tab_opt, tab_emission = st.tabs(['Contrast Analysis', 'Filter Optimizer', 'High-Energy Fluence Optimizer'])
        
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
                
                ccols = st.columns(4)
                ccols[0].metric(f'Signal through A ({target_a_mat})', f"{format_sci(contrast_res['signal_A'])} {signal_unit}")
                ccols[1].metric(f'Signal through B ({target_b_mat})', f"{format_sci(contrast_res['signal_B'])} {signal_unit}")
                ccols[2].metric('Weber Contrast |A-B|/A', f"{contrast_res['contrast']*100:.2f} %")
                ccols[3].metric('CNR / √exposure', f"{format_sci(contrast_res['cnr_per_sqrt_exposure'])}",
                                help='Compound-Poisson signal difference divided by sqrt(variance_A + variance_B) per unit exposure.')
                
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
                    yaxis_title=f'Deposited energy density ({signal_unit}/keV)',
                    xaxis=dict(range=[0, float(np.nanmax(df_A['Energy_KeV'].values)) * 1.02]),
                    legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_ct, width='stretch')
            except Exception as e:
                st.error(f"Contrast calculation failed: {e}")
                
        with tab_opt:
            st.caption("Automatically find the best filter to penetrate shadowing material while maximizing Contrast-to-Noise Ratio (CNR).")
            st.caption('Normalized CNR uses independent A/B ray noise and a fixed detector energy capacity. Multiply by the square root of capacity in keV to obtain the idealized CNR. This primary-ray search includes the selected air path but excludes filter secondaries and object scatter. Detector response assumes local first-interaction deposition; escape/reabsorption fluctuations and optical/electronic noise need detector-specific validation.')
            
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
            opt_chk1, opt_chk2 = st.columns(2)
            opt_stack_mode = opt_chk1.checkbox(
                'Optimize Additional Layer on Current Filter Stack',
                value=False,
                key='opt_stack_mode',
                help='When checked, pre-attenuates the primary spectrum through your current Filter Stack before optimizing an additional layer, and appends the chosen filter instead of replacing the stack.'
            )
            opt_include_fog = opt_chk2.checkbox(
                'Include Filter Secondary Fog (Fluorescence + Scatter) in CNR Noise',
                value=False,
                key='opt_include_fog',
                help='When checked, includes unmodulated K-fluorescence and Compton/Rayleigh scatter emitted by the candidate filter within the acceptance cone in both the air-path detector load and the compound-Poisson variance.'
            )
            active_pre_filters = (
                [{'material': f['material'], 'thickness': float(f['thickness'])} for f in st.session_state.filters]
                if opt_stack_mode and len(st.session_state.filters) > 0
                else []
            )
            
            if optimize_btn:
                st.info("Running Penetration vs Contrast Filter Optimizer...")
                try:
                    energy, fluence = load_active_input_spectrum()

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

                    raw_photons_for_opt = fluence * np.exp(-attenuation_exponent('Air', energy, distance_to_target))
                    for pf in active_pre_filters:
                        raw_photons_for_opt = raw_photons_for_opt * np.exp(
                            -attenuation_exponent(pf['material'], energy, pf['thickness'])
                        )
                                
                    df_opt_res, top_opt_res = ct_optimizer.run_filter_optimization(
                        energies=energy,
                        raw_photons=raw_photons_for_opt,
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
                        filter_materials=list(st.session_state.enabled_materials),
                        include_secondary_fog=opt_include_fog,
                        geom_params={'beam_angle_deg': beam_angle},
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
                        'pre_filters': active_pre_filters,
                        'include_secondary_fog': opt_include_fog,
                        'beam_angle': beam_angle,
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
                        'uploaded_file_name': uploaded_file.name if uploaded_file else None,
                        'spekpy_kvp': spekpy_kvp,
                        'spekpy_th': spekpy_th,
                        'spekpy_targ': spekpy_targ,
                        'spekpy_dk': spekpy_dk,
                        'tube_window': tube_window, 'tube_current_ma': tube_current_ma,
                        'source_distance_mm': distance_to_target
                    }
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    
            if st.session_state.filter_opt_results is not None:
                res = st.session_state.filter_opt_results
                
                # Check if filter optimizer results are stale
                filter_opt_stale = False
                if (res.get('candidates') != st.session_state.enabled_materials or
                    res.get('pre_filters', []) != active_pre_filters or
                    res.get('include_secondary_fog', False) != opt_include_fog or
                    (opt_include_fog and res.get('beam_angle') != beam_angle) or
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
                    res.get('uploaded_file_name') != (uploaded_file.name if uploaded_file else None) or
                    res['spekpy_kvp'] != spekpy_kvp or
                    res['spekpy_th'] != spekpy_th or
                    res['spekpy_targ'] != spekpy_targ or
                    res.get('spekpy_dk') != spekpy_dk or
                    res['tube_window'] != tube_window or
                    res.get('tube_current_ma') != tube_current_ma or
                    res.get('source_distance_mm') != distance_to_target):
                    filter_opt_stale = True
                    
                if filter_opt_stale:
                    st.warning("⚠️ **Optimization results are out of date.** Settings have changed. Click 'Run Filter Optimizer' to update.")
                    
                df_opt_res = res['df_opt_res']
                top_opt_res = res['top_opt_res']
                used_pre_filters = res.get('pre_filters', [])
                used_fog = res.get('include_secondary_fog', False)
                
                st.subheader('Optimization Results')
                if used_pre_filters:
                    pre_desc = " → ".join(f"{f['material']} ({f['thickness']:.2f} mm)" for f in used_pre_filters)
                    st.write(f"Top additional filter thicknesses behind `{pre_desc}` (Max {opt_time_limit}x additional Time Penalty):")
                else:
                    st.write(f"Top filter thicknesses balancing Dynamic Contrast against detector saturation limits (Max {opt_time_limit}x Time Penalty):")
                
                opt_display_cols = ['Plot', 'Filter_Material', 'Thickness_mm', 'Dynamic_CNR', 'Time_Penalty']
                opt_disabled_cols = ['Filter_Material', 'Thickness_mm', 'Dynamic_CNR', 'Time_Penalty']
                if used_fog and 'Secondary_Fog_Pct' in top_opt_res.columns:
                    opt_display_cols.append('Secondary_Fog_Pct')
                    opt_disabled_cols.append('Secondary_Fog_Pct')
                edited_top_opt_res = st.data_editor(
                    top_opt_res[opt_display_cols],
                    column_config={
                        'Plot': st.column_config.CheckboxColumn('Plot', help="Select to display on the chart below"),
                        'Filter_Material': 'Material',
                        'Thickness_mm': st.column_config.NumberColumn('Thickness (mm)', format='%.2f'),
                        'Dynamic_CNR': st.column_config.NumberColumn('Normalized CNR', format='%.8f'),
                        'Time_Penalty': st.column_config.NumberColumn('Time Penalty', format='%.1fx'),
                        'Secondary_Fog_Pct': st.column_config.NumberColumn('Secondary Fog (%)', format='%.2f %%'),
                    },
                    disabled=opt_disabled_cols,
                    hide_index=False,
                    width='stretch',
                    key='opt_data_editor'
                )
                st.markdown("### 📥 Apply Optimized Filter")
                if used_pre_filters:
                    st.write("Append an optimized filter layer to your pre-filtered stack:")
                else:
                    st.write("Set an optimized filter as your active filter stack (automatically enables fluorescence analysis):")
                cols_apply = st.columns(4)
                for idx, (_, row) in enumerate(edited_top_opt_res.iterrows()):
                    mat = row['Filter_Material']
                    thick = row['Thickness_mm']
                    col_idx = idx % 4
                    if cols_apply[col_idx].button(f"Apply {mat} ({thick:.2f} mm)", key=f"apply_opt_{mat}", width='stretch'):
                        new_layer = {'material': mat, 'thickness': round(float(thick), 3)}
                        if used_pre_filters:
                            st.session_state.filters = [f.copy() for f in used_pre_filters] + [new_layer]
                        else:
                            st.session_state.filters = [new_layer]
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
                    title='Time Penalty vs Normalized CNR at fixed detector capacity',
                    legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig_opt.update_yaxes(title_text="CNR / sqrt(capacity in keV)", secondary_y=False)
                fig_opt.update_yaxes(title_text="Filter Thickness (mm)", secondary_y=True)
                fig_opt.update_xaxes(title_text="Time Penalty (x)", range=[1.0, 5.0])
                
                st.plotly_chart(fig_opt, width='stretch')
        with tab_emission:
            st.markdown("### High-Energy Fluence Optimizer")
            st.write('Find filters that remove energy fluence below a cutoff threshold (default 57 keV for W K-lines) while preserving energy fluence at and above the threshold, including the continuum. This primary-transmission estimate excludes filter fluorescence and scatter; check candidates in the full stack calculator (up to 300 keV).')
            st.caption('Standalone tungsten source: 12° anode, 0.1-keV bins, no tube-window or air filtration. These settings are independent of the main calculator.')
            
            e_col1, e_col2, e_col3 = st.columns(3)
            opt_kvp = e_col1.number_input("Tube Voltage (kVp)", min_value=60, max_value=450, value=220, step=10, key='opt_em_kvp')
            opt_max_t = e_col2.number_input("Max Thickness (mm)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key='opt_em_thick')
            opt_em_thresh = e_col3.number_input(
                "Energy Threshold (keV)",
                min_value=5.0,
                max_value=float(opt_kvp - 1),
                value=min(57.0, float(opt_kvp - 1)),
                step=1.0,
                key='opt_em_thresh',
                help="Cutoff separating the low-energy band to suppress from the high-energy band to preserve (57 keV corresponds to W K-alpha lines)."
            )
            opt_em_enabled_only = st.checkbox(
                "Only Sweep Enabled Materials (⚙️ Materials)",
                value=False,
                key="opt_em_enabled_only",
                help="When checked, restricts the sweep to materials checked in the ⚙️ Materials popover instead of all 36 practical filter elements."
            )
            em_materials = list(st.session_state.enabled_materials) if opt_em_enabled_only else None
            
            if st.button("Run High-Energy Optimizer", type="primary"):
                with st.spinner("Sweeping practical filter materials..."):
                    try:
                        from emission_optimizer import run_emission_optimizer
                        df_em = run_emission_optimizer(
                            kvp=opt_kvp,
                            max_thickness_mm=opt_max_t,
                            threshold_kev=opt_em_thresh,
                            filter_materials=em_materials,
                        )
                        st.session_state.emission_opt_results = df_em
                        st.session_state.emission_opt_kvp = opt_kvp
                        st.session_state.emission_opt_max_t = opt_max_t
                        st.session_state.emission_opt_thresh = opt_em_thresh
                        st.session_state.emission_opt_materials = em_materials
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
            
            if 'emission_opt_results' in st.session_state and st.session_state.emission_opt_results is not None:
                if (st.session_state.emission_opt_kvp != opt_kvp or
                    st.session_state.emission_opt_max_t != opt_max_t or
                    st.session_state.get('emission_opt_thresh', 57.0) != opt_em_thresh or
                    st.session_state.get('emission_opt_materials') != em_materials):
                    st.warning("⚠️ **Settings have changed.** Click 'Run High-Energy Optimizer' to update results.")
                    
                df_em = st.session_state.emission_opt_results
                thresh_used = st.session_state.get('emission_opt_thresh', 57.0)
                
                st.write("Best thickness for each material, ranked by Score = (Removed Low %) * (Kept High %)^2")
                edited_df_em = st.data_editor(
                    df_em.head(50),
                    column_config={
                        'Filter_Material': 'Material',
                        'Thickness_mm': st.column_config.NumberColumn('Thickness (mm)', format='%.2f'),
                        'Removed_Low_Pct': st.column_config.NumberColumn(f'Removed <{thresh_used:g} keV (%)', format='%.2f %%'),
                        'Kept_High_Pct': st.column_config.NumberColumn(f'Kept >={thresh_used:g} keV (%)', format='%.2f %%'),
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
