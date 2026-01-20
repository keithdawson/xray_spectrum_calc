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
    sym_to_z = {"Mo":42, "Pb":82, "Cu":29, "W":74, "Ag":47, "Al":13, "N":7, "O":8, "Ar":18}
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
    # 1. Geometry factor (Solid Angle Fraction)
    # Omega / 4pi = (1 - cos(theta)) / 2
    theta = math.atan2(geom_params['radius_mm'], geom_params['distance_mm'])
    frac_accept = (1.0 - math.cos(theta)) / 2.0
    
    # 2. Material Properties
    if material == "Air":
        density = DENSITIES["Air"]
        mu_tot, _ = get_mixture_attenuation(E_in, AIR_COMPOSITION, att_df)
        # We typically skip fluorescence for Air in radiography as it's negligible/isotropic noise
        # unless specific K-lines of Ar are required.
        tauK_eff = np.zeros_like(E_in) 
        fluorescence_yield = 0.0
        calc_fluorescence = False 
    else:
        z, sym = get_element_properties(material)
        density = DENSITIES.get(sym, 1.0)
        
        # Get Attenuation
        elem_att = att_df[att_df['Element'] == z]
        mu_tot = np.interp(E_in, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
        
        # Get Fluorescence Data (EPDL) if needed
        # (Simplified loading logic based on your script)
        # You would load the ZAxxx_extracted.csv here to get tauK
        # For this example, we assume we have vectors or 0 if file missing
        tauK_eff = np.zeros_like(E_in) # Placeholder: Load your EPDL tauK here
        
        # Determine Omega and K-Energies (Values from your script or lookups)
        # Example for Mo (Z=42)
        if z == 42: 
            fluorescence_yield = 0.765
            E_Ka, E_Kb = 17.48, 19.61
            K_ratio = 1.0 / (1.0 + 0.12) # Kalpha branching
        elif z == 82: # Pb
             fluorescence_yield = 0.963
             E_Ka, E_Kb = 74.97, 84.94
             K_ratio = 1.0 / (1.0 + 0.12)
        else:
             fluorescence_yield = 0.0 # Placeholder
             
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
        frac_K_shell = np.divide(tauK_eff, mu_tot, out=np.zeros_like(mu_tot), where=mu_tot!=0)
        
        # Rate of vacancy creation
        vacancies = phi_in * prob_abs * frac_K_shell
        
        # Total isotropic emission
        total_emission = np.trapz(vacancies, E_in) * fluorescence_yield
        
        # Beam-directed emission (Geometric fraction)
        # We also need self-absorption correction (Escape Probability) for the filter itself
        # P_esc approx (1 - exp(-mu_emit * rho * t)) / (mu_emit * rho * t)
        
        # Add to specific bins for Ka and Kb
        # (This adds the 'peaks' to the spectrum)
        # ... [Logic from your script to add to closest energy bins] ...
        
        # For simplicity in this snippet, we just return the counts to be added
        # You would distribute `total_emission * frac_accept * P_esc` into the bins
        pass 

    return phi_transmitted + phi_fluorescence

def run_stack(spectrum_csv, filter_stack, att_file):
    """
    filter_stack: list of dicts [{'material': 'Mo', 'thickness': 0.1}, {'material': 'Air', 'thickness': 1000}]
    """
    df = pd.read_csv(spectrum_csv)
    E = df['Energy_KeV'].values
    Phi = df['Photons_per_s'].values
    att_df = pd.read_csv(att_file) # Load master attenuation file
    
    print(f"Initial Photon Count: {np.sum(Phi):.2e}")

    # Distance accumulation for geometry (optional, if distance is cumulative)
    current_dist = 0 
    
    for f in filter_stack:
        mat = f['material']
        thick = f['thickness']
        
        # Assume detector is at the end of the stack? 
        # Or assumes 'distance' is from current filter to detector.
        # Let's assume a fixed detector distance defined globally or per filter relative to detector.
        geom = {'radius_mm': 10.0, 'distance_mm': 100.0} # Example
        
        print(f"Applying filter: {mat} ({thick} mm)...")
        Phi = process_single_layer(E, Phi, mat, thick, att_df, "./pyne_edpl", geom)
        
        print(f"  > Photon Count after {mat}: {np.sum(Phi):.2e}")

    # Output
    pd.DataFrame({'Energy_KeV': E, 'Photons_per_s': Phi}).to_csv("final_spectrum.csv", index=False)
# ---------------------------
# CLI
def cli():
    p = argparse.ArgumentParser(description="Apply filter + fluorescence to spectrum")
    p.add_argument('--spectrum', required=True, help='input spectrum CSV (Energy_KeV, Photons_per_s)')
    p.add_argument('--filter', required=True, help='filter element (Z number, e.g. 42, or symbol e.g. Mo)')
    p.add_argument('--thickness-mm', required=True, type=float, help='filter thickness in mm')
    p.add_argument('--edpl-folder', default='.', help='folder with EPDL-extracted CSV files')
    p.add_argument('--attenuation', default='attenuation_coefficients_long.csv', help='path to attenuation_coefficients_long.csv')
    p.add_argument('--acceptance-radius-mm', default=1.0, type=float, help='circular acceptance radius at detector (mm)')
    p.add_argument('--distance-mm', default=100.0, type=float, help='distance from filter to acceptance plane (mm)')
    p.add_argument('--kbeta_ratio', default=0.12, type=float, help='Kbeta / Kalpha intensity ratio (default 0.12)')
    p.add_argument('--out', dest='out', default='spectrum_after_filter.csv', help='output CSV path')
    args = p.parse_args()
    run(args.spectrum, args.filter, args.thickness_mm, args.edpl_folder, args.attenuation,
        args.acceptance_radius_mm, args.distance_mm, args.kbeta_ratio, args.out)

if __name__ == '__main__':
    cli()
