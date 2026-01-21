import numpy as np
import pandas as pd
import math
import sys
import os
from pathlib import Path
import subprocess

# Import from the main script
try:
    import apply_filter_and_flourescence as aff
except ImportError:
    print("Error: Could not import apply_filter_and_flourescence.py. Make sure it is in the current directory.")
    sys.exit(1)

def process_single_layer_force_forward(E_in, phi_in, material, thickness_mm, att_df, edpl_folder):
    """
    Modified version of process_single_layer that assumes 100% geometric acceptance for fluorescence.
    """
    # 0. Bin widths
    dE = np.gradient(E_in)

    # 1. Geometry factor - FORCED TO 1.0 for this test
    frac_accept = 1.0
    
    # 2. Material Properties
    z, sym = aff.get_element_properties(material)
    density = aff.DENSITIES.get(sym, 1.0)
    
    # Get Attenuation
    elem_att = att_df[att_df['Element'] == z]
    if elem_att.empty:
        print(f"Error: No attenuation data for Z={z}")
        return phi_in

    mu_tot = np.interp(E_in, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
    
    # Get Fluorescence Data (EPDL)
    tauK_eff = np.zeros_like(E_in)
    if edpl_folder:
        p = Path(edpl_folder)
        candidates = list(p.glob(f"*ZA{z:03d}*.csv"))
        if candidates:
            try:
                df_edpl = pd.read_csv(candidates[0])
                tau_col = None
                if 'tauK_over_rho_cm2_per_g' in df_edpl.columns:
                    tau_col = 'tauK_over_rho_cm2_per_g'
                else:
                    for c in df_edpl.columns:
                        if c.startswith('subshell_MT534'):
                            tau_col = c
                            break
                if 'E_keV' in df_edpl.columns and tau_col:
                    tauK_eff = np.interp(E_in, df_edpl['E_keV'], df_edpl[tau_col], left=0, right=0)
            except Exception:
                pass

    # Fluorescence Parameters (Hardcoded lookup from main script)
    fl_params = {
        13: {'yield': 0.039, 'Ka': 1.49,  'Kb': 1.55,  'Kb_ratio': 0.02},
        26: {'yield': 0.32,  'Ka': 6.40,  'Kb': 7.06,  'Kb_ratio': 0.12},
        29: {'yield': 0.44,  'Ka': 8.05,  'Kb': 8.90,  'Kb_ratio': 0.13},
        42: {'yield': 0.765, 'Ka': 17.48, 'Kb': 19.61, 'Kb_ratio': 0.17},
        47: {'yield': 0.83,  'Ka': 22.16, 'Kb': 24.94, 'Kb_ratio': 0.18},
        50: {'yield': 0.84,  'Ka': 25.27, 'Kb': 28.48, 'Kb_ratio': 0.19},
        74: {'yield': 0.94,  'Ka': 59.32, 'Kb': 67.24, 'Kb_ratio': 0.21},
        79: {'yield': 0.96,  'Ka': 68.80, 'Kb': 77.98, 'Kb_ratio': 0.22},
        82: {'yield': 0.963, 'Ka': 74.97, 'Kb': 84.94, 'Kb_ratio': 0.22},
    }
    
    if z in fl_params:
        f_data = fl_params[z]
        fluorescence_yield = f_data['yield']
        E_Ka = f_data['Ka']
        E_Kb = f_data['Kb']
        kb_ratio = f_data.get('Kb_ratio', 0.12)
    else:
        fluorescence_yield = 0.0
        E_Ka, E_Kb = 0.0, 0.0
        kb_ratio = 0.12
             
    # 3. Attenuation Calculation
    t_cm = thickness_mm / 10.0
    rho_t = density * t_cm
    
    transmission_factor = np.exp(-mu_tot * rho_t)
    phi_transmitted = phi_in * transmission_factor
    
    # 4. Fluorescence Calculation
    phi_fluorescence = np.zeros_like(phi_in)
    
    if fluorescence_yield > 0:
        prob_abs = 1.0 - transmission_factor
        
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_K_shell = np.where(mu_tot > 0, tauK_eff / mu_tot, 0.0)
        
        vacancies = phi_in * prob_abs * frac_K_shell
        
        trapz = getattr(np, 'trapezoid', np.trapz)
        total_emission = trapz(vacancies, E_in) * fluorescence_yield
        
        if total_emission > 0:
            br_ka = 1.0 / (1.0 + kb_ratio)
            br_kb = 1.0 - br_ka
            
            # Self-absorption
            mu_Ka = np.interp(E_Ka, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
            mu_Kb = np.interp(E_Kb, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
            
            def get_esc(mu, rt):
                val = mu * rt
                if val < 1e-5: return 1.0
                return (1.0 - np.exp(-val)) / val
            
            esc_Ka = get_esc(mu_Ka, rho_t)
            esc_Kb = get_esc(mu_Kb, rho_t)
            
            idx_Ka = (np.abs(E_in - E_Ka)).argmin()
            idx_Kb = (np.abs(E_in - E_Kb)).argmin()
            
            # Add to bins
            phi_fluorescence[idx_Ka] += (total_emission * br_ka * frac_accept * esc_Ka) / dE[idx_Ka]
            phi_fluorescence[idx_Kb] += (total_emission * br_kb * frac_accept * esc_Kb) / dE[idx_Kb]
            
            print(f"  [Test] Total Emission: {total_emission:.2e}")
            print(f"  [Test] Added to Ka ({E_Ka} keV): {(total_emission * br_ka * frac_accept * esc_Ka):.2e}")
            print(f"  [Test] Added to Kb ({E_Kb} keV): {(total_emission * br_kb * frac_accept * esc_Kb):.2e}")

    return phi_transmitted + phi_fluorescence

def main():
    # Configuration
    spectrum_file = '225_W_spekpy_output.csv'
    
    # Try to find attenuation file
    possible_att_files = [
        'attenuation_coefficients_long.csv',
        'extract_atten_coeff/attenuation_coefficients_long.csv'
    ]
    attenuation_file = None
    for f in possible_att_files:
        if os.path.exists(f):
            attenuation_file = f
            break
    
    if not attenuation_file:
        print("Error: Could not find attenuation_coefficients_long.csv")
        sys.exit(1)

    edpl_folder = 'pyne_edpl'
    output_file = 'test_fluorescence_output.csv'
    
    # Filter settings for test
    material = 'Mo'
    thickness = 0.5 # mm
    
    print(f"Loading spectrum from {spectrum_file}...")
    try:
        df = pd.read_csv(spectrum_file)
        # Handle column names
        if 'Energy_keV' in df.columns:
            E = df['Energy_keV'].values
        elif 'Energy_KeV' in df.columns:
            E = df['Energy_KeV'].values
        else:
            E = df.iloc[:, 0].values
            
        if 'Relative_Intensity' in df.columns:
            Phi = df['Relative_Intensity'].values
        elif 'Photons_per_s' in df.columns:
            Phi = df['Photons_per_s'].values
        else:
            Phi = df.iloc[:, 1].values
            
    except Exception as e:
        print(f"Error reading spectrum file: {e}")
        sys.exit(1)
    
    print(f"Loading attenuation from {attenuation_file}...")
    att_df = pd.read_csv(attenuation_file)
    if 'Energy_MeV' in att_df.columns and 'Energy_KeV' not in att_df.columns:
        att_df['Energy_KeV'] = att_df['Energy_MeV'] * 1000
        
    print(f"Running Forced Forward Fluorescence Test for {material} ({thickness} mm)...")
    Phi_out = process_single_layer_force_forward(E, Phi, material, thickness, att_df, edpl_folder)
    
    # Save
    out_df = pd.DataFrame({'Energy_KeV': E, 'Photons_per_s': Phi_out})
    out_df.to_csv(output_file, index=False)
    print(f"Saved output to {output_file}")
    
    # Plot
    print("Plotting comparison...")
    subprocess.run([sys.executable, 'plot_csv.py', 'filtered_spectrum.csv', output_file])

if __name__ == "__main__":
    main()