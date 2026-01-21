#!/usr/bin/env python3
"""
apply_filter_and_fluorescence.py

Usage:
  python apply_filter_and_fluorescence.py \
      --spectrum w_tungsten_225kv_spectrum.csv \
      --filter Mo \
      --thickness-mm 0.10 \
      [--edpl-folder ./] \
      [--attenuation attenuation_coefficients_long.csv] \
      [--acceptance-radius-mm 1.0] [--distance-mm 100.0] \
      [--kbeta_ratio 0.12] \
      [--out out_spectrum_after_filter.csv]

Description:
  Reads input spectrum (Energy_KeV, Photons_per_s). Uses attenuation_coefficients_long.csv
  (Energy_KeV, Element, Mu_over_rho, Mu_en_over_rho) and the EPDL-extracted CSV for the
  chosen element (which must contain columns E_keV, mu_over_rho_cm2_per_g, tauK_over_rho_cm2_per_g)
  to calculate the transmitted photon spectrum through the filter and add fluorescence
  (K-alpha and K-beta) emitted by the filter that are traveling mostly along the same path
  as the incident beam.

  The acceptance of fluoresced photons into the "same path" is modeled as a cone defined by
  acceptance_radius_mm at a distance distance_mm. The fraction of isotropic emission inside
  that cone is: fraction = (1 - cos(theta)) / 2 with theta = arctan(radius / distance).

Notes:
  - Script expects mass attenuation coefficients in cm^2/g and thickness in mm.
  - Needs a density mapping for the element (small built-in table includes common metals).
  - If an EPDL/extracted file can't be found, the script will error with suggestions.
"""

import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ---------------------------
# small density lookup (add other elements if you need)

# --- Constants & Lookup ---
DENSITIES = {
    "Mo": 10.28,
    "Pb": 11.34,
    "Cu": 8.96,
    "Al": 2.70,
    "Fe": 7.87,
    "W": 19.25,
    "Ag": 10.49,
    "Au": 19.32,
    "Sn": 7.31,
    "Air": 0.001225  # g/cm^3 at sea level
}

AIR_COMPOSITION = {
    7: 0.755,  # Nitrogen
    8: 0.232,  # Oxygen
    18: 0.013  # Argon (Approximation)
}

def get_element_properties(symbol_or_z):
    """Returns Z, Symbol for a given input."""
    # Maps for common elements
    sym_to_z = {"Mo":42, "Pb":82, "Cu":29, "W":74, "Ag":47, "Al":13, "N":7, "O":8, "Ar":18, "Fe":26, "Au":79, "Sn":50}
    z_to_sym = {v: k for k, v in sym_to_z.items()}
    
    if str(symbol_or_z).isdigit():
        z = int(symbol_or_z)
        return z, z_to_sym.get(z, "Unknown")
    else:
        sym = str(symbol_or_z).capitalize()
        return sym_to_z.get(sym, None), sym

def load_data(edpl_folder, attenuation_csv):
    """Loads EPDL files and attenuation master list."""
    att_df = pd.read_csv(attenuation_csv)
    # Cache EPDL files as needed to avoid re-reading
    return att_df

def get_mixture_attenuation(energy_axis, components, att_df):
    """
    Computes effective mu/rho for a mixture.
    components: dict {Z: weight_fraction}
    """
    mu_mix = np.zeros_like(energy_axis)
    mu_en_mix = np.zeros_like(energy_axis)
    mu_mix = np.zeros_like(energy_axis, dtype=float)
    mu_en_mix = np.zeros_like(energy_axis, dtype=float)
    
    for z, weight in components.items():
        elem_df = att_df[att_df['Element'] == z]
        if elem_df.empty:
            raise ValueError(f"Missing data for Z={z} in attenuation file")
        
        # Interp to spectrum energy axis
        mu = np.interp(energy_axis, elem_df['Energy_KeV'], elem_df['Mu_over_rho'])
        mu_en = np.interp(energy_axis, elem_df['Energy_KeV'], elem_df['Mu_en_over_rho'])
        
        mu_mix += mu * weight
        mu_en_mix += mu_en * weight
        
    return mu_mix, mu_en_mix

def process_single_layer(E_in, phi_in, material, thickness_mm, att_df, edpl_folder, 
                        geom_params, calc_fluorescence=True):
    """
    Processes one layer of material.
    geom_params: dict with 'radius_mm', 'distance_mm'
    """
    # 0. Bin widths for integration/density conversion
    dE = np.gradient(E_in)

    # 1. Geometry factor (Solid Angle Fraction)
    # Omega / 4pi = (1 - cos(theta)) / 2
    theta = math.atan2(geom_params['radius_mm'], geom_params['distance_mm'])
    frac_accept = (1.0 - math.cos(theta)) / 2.0
    
    # 2. Material Properties
    if material.lower() == "air":
        density = DENSITIES["Air"]
        mu_tot, _ = get_mixture_attenuation(E_in, AIR_COMPOSITION, att_df)
        # We typically skip fluorescence for Air in radiography as it's negligible/isotropic noise
        # unless specific K-lines of Ar are required.
        tauK_eff = np.zeros_like(E_in) 
        fluorescence_yield = 0.0
        elem_att = None
        calc_fluorescence = False 
    else:
        z, sym = get_element_properties(material)
        if z is None:
            raise ValueError(f"Unknown material or element: '{material}'. Please check spelling or update lookup tables.")
        density = DENSITIES.get(sym, 1.0)
        
        # Get Attenuation
        elem_att = att_df[att_df['Element'] == z]
        mu_tot = np.interp(E_in, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
        
        # Get Fluorescence Data (EPDL) if needed
        tauK_eff = np.zeros_like(E_in)
        if edpl_folder:
            # Try to find file ZA{z:03d}*.csv
            p = Path(edpl_folder)
            candidates = list(p.glob(f"*ZA{z:03d}*.csv"))
            if candidates:
                try:
                    df_edpl = pd.read_csv(candidates[0])
                    # Check for required columns
                    tau_col = None
                    if 'tauK_over_rho_cm2_per_g' in df_edpl.columns:
                        tau_col = 'tauK_over_rho_cm2_per_g'
                    else:
                        # Look for subshell_MT534 (K-shell)
                        for c in df_edpl.columns:
                            if c.startswith('subshell_MT534'):
                                tau_col = c
                                break

                    if 'E_keV' in df_edpl.columns and tau_col:
                        tauK_eff = np.interp(E_in, df_edpl['E_keV'], df_edpl[tau_col], left=0, right=0)
                    else:
                        print(f"Warning: EPDL file for Z={z} ({candidates[0]}) missing required columns.")
                except Exception as e:
                    print(f"Warning: Failed to parse EPDL file for Z={z}: {e}")
            else:
                print(f"Warning: No EPDL data found for Z={z} in '{edpl_folder}'. Fluorescence calculation will be inaccurate (tauK=0).")

        # Fluorescence Parameters
        fl_params = {
            13: {'yield': 0.039, 'Ka': 1.49,  'Kb': 1.55,  'Kb_ratio': 0.02}, # Al
            26: {'yield': 0.32,  'Ka': 6.40,  'Kb': 7.06,  'Kb_ratio': 0.12}, # Fe
            29: {'yield': 0.44,  'Ka': 8.05,  'Kb': 8.90,  'Kb_ratio': 0.13}, # Cu
            42: {'yield': 0.765, 'Ka': 17.48, 'Kb': 19.61, 'Kb_ratio': 0.17}, # Mo
            47: {'yield': 0.83,  'Ka': 22.16, 'Kb': 24.94, 'Kb_ratio': 0.18}, # Ag
            50: {'yield': 0.84,  'Ka': 25.27, 'Kb': 28.48, 'Kb_ratio': 0.19}, # Sn
            74: {'yield': 0.94,  'Ka': 59.32, 'Kb': 67.24, 'Kb_ratio': 0.21}, # W
            79: {'yield': 0.96,  'Ka': 68.80, 'Kb': 77.98, 'Kb_ratio': 0.22}, # Au
            82: {'yield': 0.963, 'Ka': 74.97, 'Kb': 84.94, 'Kb_ratio': 0.22}, # Pb
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
    
    # Transmission: I = I0 * exp(-mu * rho * t)
    transmission_factor = np.exp(-mu_tot * rho_t)
    phi_transmitted = phi_in * transmission_factor
    
    # 4. Fluorescence Calculation
    phi_fluorescence = np.zeros_like(phi_in)
    
    if calc_fluorescence and fluorescence_yield > 0:
        # P_absorption = 1 - exp(-mu * rho * t)
        prob_abs = 1.0 - transmission_factor
        
        # Fraction of interactions that are photoelectric K-shell
        # Note: Avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_K_shell = np.where(mu_tot > 0, tauK_eff / mu_tot, 0.0)
        
        # Rate of vacancy creation
        vacancies = phi_in * prob_abs * frac_K_shell
        
        # Total isotropic emission
        trapz = getattr(np, 'trapezoid', np.trapz)
        total_emission = trapz(vacancies, E_in) * fluorescence_yield
        
        # Beam-directed emission (Geometric fraction)
        if total_emission > 0:
            # K-alpha / K-beta split
            br_ka = 1.0 / (1.0 + kb_ratio)
            br_kb = 1.0 - br_ka
            
            # Self-absorption (Escape Probability)
            mu_Ka = np.interp(E_Ka, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
            mu_Kb = np.interp(E_Kb, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
            
            def get_esc(mu, rt):
                val = mu * rt
                if val < 1e-5: return 1.0
                return (1.0 - np.exp(-val)) / val
            
            esc_Ka = get_esc(mu_Ka, rho_t)
            esc_Kb = get_esc(mu_Kb, rho_t)
            
            # Add to bins (convert total counts to density by dividing by dE)
            idx_Ka = (np.abs(E_in - E_Ka)).argmin()
            idx_Kb = (np.abs(E_in - E_Kb)).argmin()
            
            phi_fluorescence[idx_Ka] += (total_emission * br_ka * frac_accept * esc_Ka) / dE[idx_Ka]
            phi_fluorescence[idx_Kb] += (total_emission * br_kb * frac_accept * esc_Kb) / dE[idx_Kb]

    return phi_transmitted + phi_fluorescence

def run_stack(spectrum_csv, filter_stack, att_file, edpl_folder, geom_params, out_path):
    """
    Processes a spectrum through a stack of filters.

    Args:
        spectrum_csv (str): Path to the input spectrum CSV.
        filter_stack (list): List of filter dictionaries, e.g., [{'material': 'Mo', 'thickness': 0.1}].
        att_file (str): Path to the master attenuation data file.
        edpl_folder (str): Path to the folder with EPDL-extracted data.
        geom_params (dict): Dictionary with geometry parameters ('radius_mm', 'distance_mm').
        out_path (str): Path for the output CSV file.
    """
    try:
        df = pd.read_csv(spectrum_csv)
    except FileNotFoundError:
        sys.exit(f"Error: Input spectrum file not found at '{spectrum_csv}'")

    # Be flexible with input column names
    if 'Energy_KeV' not in df.columns and 'Energy_keV' in df.columns:
        df = df.rename(columns={'Energy_keV': 'Energy_KeV'})
    if 'Photons_per_s' not in df.columns and 'Relative_Intensity' in df.columns:
        df = df.rename(columns={'Relative_Intensity': 'Photons_per_s'})

    if 'Energy_KeV' not in df.columns or 'Photons_per_s' not in df.columns:
        sys.exit("Error: Input CSV must have columns like 'Energy_KeV' and 'Photons_per_s' or 'Energy_keV' and 'Relative_Intensity'")

    E = df['Energy_KeV'].values
    Phi = df['Photons_per_s'].values
    
    try:
        att_df = pd.read_csv(att_file)
    except FileNotFoundError:
        sys.exit(f"Error: Attenuation file not found at '{att_file}'")

    # Handle energy units (MeV vs KeV) in attenuation file
    if 'Energy_MeV' in att_df.columns and 'Energy_KeV' not in att_df.columns:
        att_df['Energy_KeV'] = att_df['Energy_MeV'] * 1000
    
    dE = np.gradient(E)
    print(f"Initial Photon Count: {np.sum(Phi * dE):.2e}")

    for f in filter_stack:
        mat = f['material']
        thick = float(f['thickness'])
        
        print(f"Applying filter: {mat} ({thick} mm)...")
        Phi = process_single_layer(E, Phi, mat, thick, att_df, edpl_folder, geom_params)
        
        print(f"  > Photon Count after {mat}: {np.sum(Phi * dE):.2e}")

    # Output
    out_df = pd.DataFrame({'Energy_KeV': E, 'Photons_per_s': Phi})
    out_df.to_csv(out_path, index=False)
    print(f"Wrote final spectrum to {out_path}")

# ---------------------------
# CLI
def cli():
    p = argparse.ArgumentParser(
        description="Apply a stack of filters to an X-ray spectrum and calculate fluorescence.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
  python %(prog)s \\
    --spectrum 225_W_input.csv \\
    --filter-materials Mo Al Air \\
    --filter-thicknesses-mm 0.1 1.0 1000 \\
    --out filtered_spectrum.csv
"""
    )
    p.add_argument('--spectrum', required=True, help='Input spectrum CSV. Must have columns like (Energy_KeV, Photons_per_s).')
    p.add_argument(
        '--filter-materials',
        required=True,
        nargs='+',
        help='Space-separated list of filter materials to apply, in order (e.g., Mo Al Air).'
    )
    p.add_argument(
        '--filter-thicknesses-mm',
        required=True,
        nargs='+',
        type=float,
        help='Space-separated list of filter thicknesses in mm, matching the order of materials.'
    )
    p.add_argument('--attenuation', default='attenuation_coefficients_long.csv', help='Path to master attenuation data file.')
    p.add_argument('--edpl-folder', default='pyne_edpl', help='Folder with EPDL-extracted CSV files for fluorescence.')
    
    # Geometry args
    p.add_argument('--acceptance-radius-mm', type=float, default=10.0, help='Acceptance cone radius at detector (mm).')
    p.add_argument('--distance-mm', type=float, default=100.0, help='Distance from the filter stack to the detector (mm).')

    p.add_argument('--out', default='final_spectrum.csv', help='Output CSV file path.')
    
    args = p.parse_args()

    # --- Parse Filters ---
    if len(args.filter_materials) != len(args.filter_thicknesses_mm):
        sys.exit("Error: The number of materials and thicknesses must be the same.")

    filter_stack = []
    for material, thickness in zip(args.filter_materials, args.filter_thicknesses_mm):
        filter_stack.append({'material': material, 'thickness': thickness})

    # --- Geom Params ---
    geom_params = {
        'radius_mm': args.acceptance_radius_mm,
        'distance_mm': args.distance_mm
    }
    
    # --- Run ---
    run_stack(
        spectrum_csv=args.spectrum,
        filter_stack=filter_stack,
        att_file=args.attenuation,
        edpl_folder=args.edpl_folder,
        geom_params=geom_params,
        out_path=args.out
    )

if __name__ == '__main__':
    cli()
