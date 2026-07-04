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
      [--beam-angle-deg 30.0] \
      [--kbeta_ratio 0.12] \
      [--out out_spectrum_after_filter.csv]

Description:
  Reads input spectrum (Energy_KeV, Photons_per_s). Uses attenuation_coefficients_long.csv
  (Energy_KeV, Element, Mu_over_rho, Mu_en_over_rho) and the EPDL-extracted CSV for the
  chosen element (which must contain columns E_keV, mu_over_rho_cm2_per_g, tauK_over_rho_cm2_per_g)
  to calculate the transmitted photon spectrum through the filter and add fluorescence
  (K-alpha and K-beta) emitted by the filter that are traveling mostly along the same path
  as the incident beam.

  The acceptance of fluoresced photons into the "same path" is modeled based on the
  source beam cone angle. The fraction of isotropic emission inside that cone is:
  fraction = (1 - cos(theta)) / 2 with theta = beam_angle / 2.

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

from fluorescence_k_data import FLUOR_K, ATOMIC_MASS

AVOGADRO = 6.02214076e23

# ---------------------------
# small density lookup (add other elements if you need)

# --- Constants & Lookup ---
DENSITIES = {
    "H": 8.375e-05,
    "He": 0.0001663,
    "Li": 0.534,
    "Be": 1.848,
    "B": 2.37,
    "C": 1.7,
    "N": 0.001165,
    "O": 0.001332,
    "F": 0.00158,
    "Ne": 0.0008385,
    "Na": 0.971,
    "Mg": 1.74,
    "Al": 2.699,
    "Si": 2.33,
    "P": 2.2,
    "S": 2.0,
    "Cl": 0.002995,
    "Ar": 0.001662,
    "K": 0.862,
    "Ca": 1.55,
    "Sc": 2.989,
    "Ti": 4.54,
    "V": 6.11,
    "Cr": 7.18,
    "Mn": 7.44,
    "Fe": 7.874,
    "Co": 8.9,
    "Ni": 8.902,
    "Cu": 8.96,
    "Zn": 7.133,
    "Ga": 5.904,
    "Ge": 5.323,
    "As": 5.73,
    "Se": 4.5,
    "Br": 0.007072,
    "Kr": 0.003478,
    "Rb": 1.532,
    "Sr": 2.54,
    "Y": 4.469,
    "Zr": 6.506,
    "Nb": 8.57,
    "Mo": 10.22,
    "Tc": 11.5,
    "Ru": 12.41,
    "Rh": 12.41,
    "Pd": 12.02,
    "Ag": 10.5,
    "Cd": 8.65,
    "In": 7.31,
    "Sn": 7.31,
    "Sb": 6.691,
    "Te": 6.24,
    "I": 4.93,
    "Xe": 0.005485,
    "Cs": 1.873,
    "Ba": 3.5,
    "La": 6.154,
    "Ce": 6.657,
    "Pr": 6.71,
    "Nd": 6.9,
    "Pm": 7.22,
    "Sm": 7.46,
    "Eu": 5.243,
    "Gd": 7.9,
    "Tb": 8.229,
    "Dy": 8.55,
    "Ho": 8.795,
    "Er": 9.066,
    "Tm": 9.321,
    "Yb": 6.73,
    "Lu": 9.84,
    "Hf": 13.31,
    "Ta": 16.65,
    "W": 19.3,
    "Re": 21.02,
    "Os": 22.57,
    "Ir": 22.42,
    "Pt": 21.45,
    "Au": 19.32,
    "Hg": 13.55,
    "Tl": 11.72,
    "Pb": 11.35,
    "Bi": 9.747,
    "Po": 9.32,
    "At": 10.0,
    "Rn": 0.009066,
    "Fr": 10.0,
    "Ra": 5.0,
    "Ac": 10.07,
    "Th": 11.72,
    "Pa": 15.37,
    "U": 18.95,
    "Air": 0.001225
}

AIR_COMPOSITION = {
    7: 0.755,  # Nitrogen
    8: 0.232,  # Oxygen
    18: 0.013  # Argon (Approximation)
}

SYM_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,
}

# Elements commercially available as thin foils/sheets/discs suitable for
# physical beam filters. Excludes gases, liquids, reactive alkali/alkaline
# earth metals, brittle metalloids (Si, Ge, Sb, Te), metals too brittle to
# roll into foil (Cr, Mn, Ru, Os, Ir), highly toxic soft metals (Tl), and
# radioactive elements. Rare earths Sm-Yb are included since they are sold
# as foils and commonly used as K-edge filters (K-edges 47-63 keV).
PRACTICAL_FILTER_MATERIALS = [
    'Be', 'C', 'Mg', 'Al', 'Ti', 'V', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Zr', 'Nb', 'Mo', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Hf', 'Ta', 'W', 'Re', 'Pt', 'Au', 'Pb', 'Bi',
]

def get_element_properties(symbol_or_z):
    """Returns Z, Symbol for a given input."""
    z_to_sym = {v: k for k, v in SYM_TO_Z.items()}
    
    if str(symbol_or_z).isdigit():
        z = int(symbol_or_z)
        return z, z_to_sym.get(z, "Unknown")
    else:
        sym = str(symbol_or_z).capitalize()
        return SYM_TO_Z.get(sym, None), sym

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

def calculate_fluorescence_escape(mu_in, mu_out, rho_t):
    """Depth-integrated escape factor for secondary photons generated
    uniformly along the beam path: integral over mass depth x of
    exp(-mu_in*x) * exp(-mu_out*(rho_t - x)) dx, in g/cm^2. mu_in is the
    attenuation at the exciting energy, mu_out at the outgoing energy."""
    mu_in = np.asarray(mu_in, dtype=float)
    denom = mu_out - mu_in
    mask = np.isclose(denom, 0.0, atol=1e-7)
    res = np.zeros_like(mu_in)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (np.exp(-mu_in * rho_t) - np.exp(-mu_out * rho_t)) / denom
        res[~mask] = np.broadcast_to(term, res.shape)[~mask]
    if np.any(mask):
        res[mask] = rho_t * np.exp(-mu_in[mask] * rho_t)
    return res

def klein_nishina_incoherent_mu(E_keV, z):
    """Free-electron Klein-Nishina incoherent mass attenuation (cm^2/g).
    Fallback when no EPDL incoherent data (MT504) is available. Accurate to
    a few percent above ~30 keV; overestimates at lower energies where
    electron binding suppresses Compton scattering."""
    k = np.asarray(E_keV, dtype=float) / 511.0
    two_pi_re2 = 4.989345e-25  # 2*pi*r_e^2 [cm^2]
    ln_term = np.log(1.0 + 2.0 * k)
    sigma_e = two_pi_re2 * (
        (1.0 + k) / k**2 * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - ln_term / k)
        + ln_term / (2.0 * k)
        - (1.0 + 3.0 * k) / (1.0 + 2.0 * k)**2
    )
    electrons_per_g = AVOGADRO * z / ATOMIC_MASS[z]
    return sigma_e * electrons_per_g

def process_single_layer(E_in, phi_in, material, thickness_mm, att_df, edpl_folder,
                        geom_params, calc_fluorescence=True, calc_scatter=True):
    """
    Processes one layer of material.
    geom_params: dict with 'radius_mm', 'distance_mm'
    """
    # 0. Bin widths for integration/density conversion
    dE = np.gradient(E_in)

    # 1. Geometry factor (Solid Angle Fraction)
    # Omega / 4pi = (1 - cos(theta)) / 2
    theta = math.radians(geom_params['beam_angle_deg'] / 2.0)
    frac_accept = (1.0 - math.cos(theta)) / 2.0
    
    # 2. Material Properties
    if material.lower() == "air":
        density = DENSITIES["Air"]
        mu_tot, _ = get_mixture_attenuation(E_in, AIR_COMPOSITION, att_df)
        # We typically skip fluorescence for Air in radiography as it's negligible/isotropic noise
        # unless specific K-lines of Ar are required. Scatter is also skipped:
        # the air column scatters along the whole beam path rather than acting
        # as a localized secondary source at the filter plane.
        tauK_eff = np.zeros_like(E_in)
        fluorescence_yield = 0.0
        elem_att = None
        mu_incoh = np.zeros_like(E_in)
        calc_fluorescence = False
        calc_scatter = False
    else:
        z, sym = get_element_properties(material)
        if z is None:
            raise ValueError(f"Unknown material or element: '{material}'. Please check spelling or update lookup tables.")
        density = DENSITIES.get(sym, 1.0)
        
        # Get Attenuation
        elem_att = att_df[att_df['Element'] == z]
        mu_tot = np.interp(E_in, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
        
        # Get Fluorescence & Scatter Data (EPDL) if needed
        tauK_eff = np.zeros_like(E_in)
        mu_incoh = None
        if edpl_folder:
            # Try to find file ZA{z:03d}*.csv
            p = Path(edpl_folder)
            candidates = list(p.glob(f"*ZA{z:03d}*.csv"))
            candidates = [c for c in candidates if 'metadata' not in c.name.lower()]
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

                    if 'E_keV' in df_edpl.columns and 'MT504' in df_edpl.columns:
                        mu_incoh = np.interp(E_in, df_edpl['E_keV'], df_edpl['MT504'])
                except Exception as e:
                    print(f"Warning: Failed to parse EPDL file for Z={z}: {e}")
            else:
                print(f"Warning: No EPDL data found for Z={z} in '{edpl_folder}'. Fluorescence calculation will be inaccurate (tauK=0).")

        if mu_incoh is None:
            mu_incoh = klein_nishina_incoherent_mu(E_in, z)

        # Fluorescence Parameters (Krause yields, rad-rate-weighted line
        # energies; see generate_fluorescence_data.py)
        if z in FLUOR_K:
            f_data = FLUOR_K[z]
            fluorescence_yield = f_data['yield']
            E_Ka = f_data['Ka']
            E_Kb = f_data['Kb']
            kb_ratio = f_data.get('Kb_ratio', 0.12)
        else:
            # Fallback empirical formulas
            # K-shell yield: approx Z^4 / (33^4 + Z^4)
            fluorescence_yield = float(z**4) / (33**4 + z**4)
            # Moseley's Law: E = 10.2 eV * (Z - 1)^2 for K-alpha
            E_Ka = 0.0102 * (z - 1)**2
            # K-beta is roughly 12-15% higher than K-alpha
            E_Kb = E_Ka * 1.12
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
        # K-alpha / K-beta split
        br_ka = 1.0 / (1.0 + kb_ratio)
        br_kb = 1.0 - br_ka
        
        # Attenuation at fluorescence energies
        mu_Ka = np.interp(E_Ka, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
        mu_Kb = np.interp(E_Kb, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])

        # Calculate transfer factors (vectorized over E_in)
        transfer_Ka = calculate_fluorescence_escape(mu_tot, mu_Ka, rho_t)
        transfer_Kb = calculate_fluorescence_escape(mu_tot, mu_Kb, rho_t)
        
        # Calculate escaping flux density (before integration)
        flux_Ka = phi_in * tauK_eff * transfer_Ka
        flux_Kb = phi_in * tauK_eff * transfer_Kb
        
        # Integrate
        trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        total_Ka = trapz(flux_Ka, E_in)
        total_Kb = trapz(flux_Kb, E_in)
        
        # Apply yield, branching, geometry
        photons_Ka = total_Ka * fluorescence_yield * br_ka * frac_accept
        photons_Kb = total_Kb * fluorescence_yield * br_kb * frac_accept
        
        # Add to bins
        idx_Ka = (np.abs(E_in - E_Ka)).argmin()
        idx_Kb = (np.abs(E_in - E_Kb)).argmin()
        
        phi_fluorescence[idx_Ka] += photons_Ka / dE[idx_Ka]
        phi_fluorescence[idx_Kb] += photons_Kb / dE[idx_Kb]

    # 5. Compton (incoherent) Scatter Fog
    # Incoherent scatter in the filter is a distributed secondary source,
    # just like fluorescence. Approximations: scattered photons keep their
    # energy (Compton loss is <16% below 100 keV for scatter angles up to
    # 90 deg) and are emitted isotropically (Klein-Nishina is actually
    # forward-peaked, so the in-cone fraction is a mild underestimate).
    phi_scatter = np.zeros_like(phi_in)
    if calc_scatter:
        transfer_scatter = calculate_fluorescence_escape(mu_tot, mu_tot, rho_t)
        phi_scatter = phi_in * mu_incoh * transfer_scatter * frac_accept

    phi_total = phi_transmitted + phi_fluorescence + phi_scatter
    return phi_total, phi_fluorescence, phi_scatter, transmission_factor

def run_stack(spectrum_csv, filter_stack, att_file, edpl_folder, geom_params, out_path, save_fluorescence=False):
    """
    Processes a spectrum through a stack of filters.

    Args:
        spectrum_csv (str): Path to the input spectrum CSV.
        filter_stack (list): List of filter dictionaries, e.g., [{'material': 'Mo', 'thickness': 0.1}].
        att_file (str): Path to the master attenuation data file.
        edpl_folder (str): Path to the folder with EPDL-extracted data.
        geom_params (dict): Dictionary with geometry parameters ('radius_mm', 'distance_mm').
        out_path (str): Path for the output CSV file.
        save_fluorescence (bool): If True, saves a separate CSV with only the fluorescence component.
    """
    try:
        df = pd.read_csv(spectrum_csv)
    except FileNotFoundError:
        raise ValueError(f"Error: Input spectrum file not found at '{spectrum_csv}'")

    # Be flexible with input column names
    if 'Energy_KeV' not in df.columns and 'Energy_keV' in df.columns:
        df = df.rename(columns={'Energy_keV': 'Energy_KeV'})
    if 'Photons_per_s' not in df.columns and 'Relative_Intensity' in df.columns:
        df = df.rename(columns={'Relative_Intensity': 'Photons_per_s'})

    if 'Energy_KeV' not in df.columns or 'Photons_per_s' not in df.columns:
        raise ValueError("Error: Input CSV must have columns like 'Energy_KeV' and 'Photons_per_s' or 'Energy_keV' and 'Relative_Intensity'")

    E = df['Energy_KeV'].values
    Phi = df['Photons_per_s'].values
    
    # Track accumulated fluorescence and scatter if requested
    Phi_fluoro_accum = np.zeros_like(Phi) if save_fluorescence else None
    Phi_scatter_accum = np.zeros_like(Phi) if save_fluorescence else None
    
    try:
        att_df = pd.read_csv(att_file)
    except FileNotFoundError:
        raise ValueError(f"Error: Attenuation file not found at '{att_file}'")

    # Handle energy units (MeV vs KeV) in attenuation file
    if 'Energy_MeV' in att_df.columns and 'Energy_KeV' not in att_df.columns:
        att_df['Energy_KeV'] = att_df['Energy_MeV'] * 1000
    
    dE = np.gradient(E)
    print(f"Initial Photon Count: {np.sum(Phi * dE):.2e}")

    for f in filter_stack:
        mat = f['material']
        thick = float(f['thickness'])
        
        print(f"Applying filter: {mat} ({thick} mm)...")
        Phi, Phi_new_fluoro, Phi_new_scatter, trans_factor = process_single_layer(E, Phi, mat, thick, att_df, edpl_folder, geom_params)

        if save_fluorescence:
            # Attenuate existing secondary components and add new ones
            Phi_fluoro_accum = Phi_fluoro_accum * trans_factor + Phi_new_fluoro
            Phi_scatter_accum = Phi_scatter_accum * trans_factor + Phi_new_scatter
        
        print(f"  > Photon Count after {mat}: {np.sum(Phi * dE):.2e}")

    # Output
    out_df = pd.DataFrame({'Energy_KeV': E, 'Photons_per_s': Phi})
    out_df.to_csv(out_path, index=False)
    print(f"Wrote final spectrum to {out_path}")
    
    if save_fluorescence:
        base = out_path[:-4] if out_path.lower().endswith('.csv') else out_path

        out_df_f = pd.DataFrame({'Energy_KeV': E, 'Photons_per_s': Phi_fluoro_accum})
        out_df_f.to_csv(base + '_fluoro.csv', index=False)
        print(f"Wrote fluorescence spectrum to {base + '_fluoro.csv'}")

        out_df_s = pd.DataFrame({'Energy_KeV': E, 'Photons_per_s': Phi_scatter_accum})
        out_df_s.to_csv(base + '_scatter.csv', index=False)
        print(f"Wrote scatter spectrum to {base + '_scatter.csv'}")

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
    p.add_argument('--beam-angle-deg', type=float, default=30.0, help='Full beam cone angle (degrees).')

    p.add_argument('--out', default='final_spectrum.csv', help='Output CSV file path.')
    p.add_argument('--save-fluorescence', action='store_true', help='Save a separate CSV with only the fluorescence component (_fluoro.csv).')
    
    args = p.parse_args()

    # --- Parse Filters ---
    if len(args.filter_materials) != len(args.filter_thicknesses_mm):
        raise ValueError("Error: The number of materials and thicknesses must be the same.")

    filter_stack = []
    for material, thickness in zip(args.filter_materials, args.filter_thicknesses_mm):
        filter_stack.append({'material': material, 'thickness': thickness})

    # --- Geom Params ---
    geom_params = {
        'radius_mm': 0.0, # Placeholder, as beam_angle_deg is now used
        'distance_mm': 0.0, # Placeholder, as beam_angle_deg is now used
        'beam_angle_deg': args.beam_angle_deg
    }
    
    # --- Run ---
    run_stack(
        spectrum_csv=args.spectrum,
        filter_stack=filter_stack,
        att_file=args.attenuation,
        edpl_folder=args.edpl_folder,
        geom_params=geom_params,
        out_path=args.out,
        save_fluorescence=args.save_fluorescence
    )

if __name__ == '__main__':
    cli()
