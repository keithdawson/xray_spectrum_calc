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
DENSITIES_G_CM3 = {
    "Mo": 10.28,
    "Pb": 11.34,
    "Cu": 8.96,
    "Al": 2.70,
    "Fe": 7.87,
    "W": 19.25,
    "Ag": 10.49,
    "Au": 19.32,
    # add others as needed...
}

# ---------------------------
# helpers
def parse_element_token(token):
    """
    Accept element token as int (Z) or string (symbol/name). Return (Z, symbol).
    If numeric string -> Z int. If symbol -> try to map to Z for common ones (Mo,Pb,...).
    """
    token = str(token).strip()
    if token.isdigit():
        Z = int(token)
        # map some common Z->symbol
        z_map = {42: "Mo", 82: "Pb", 29: "Cu", 74: "W", 79: "Au", 47: "Ag", 13: "Al", 26: "Fe"}
        sym = z_map.get(Z, None)
        return Z, sym
    else:
        # try to canonicalize symbol
        sym = token.capitalize()
        symbol_to_z = {"Mo":42, "Pb":82, "Cu":29, "W":74, "Au":79, "Ag":47, "Al":13, "Fe":26}
        Z = symbol_to_z.get(sym, None)
        return Z, sym

def find_edpl_extracted_file(edpl_folder, Z, sym):
    """
    Try to find the edpl/extracted CSV for a given element.
    Search heuristics:
      - file containing 'ZA' + zero-padded Z (e.g. ZA042 or ZA042000)
      - file containing the symbol (Mo) case-insensitive
      - file named like 'mo_mu_tau_from_epdl.csv' or 'za042_mu_tau.csv' or '*extracted.csv'
    """
    p = Path(edpl_folder)
    files = list(p.glob("*.csv")) + list(p.glob("*.CSV"))
    z_str3 = f"ZA{Z:03d}" if Z is not None else None
    candidates = []
    for f in files:
        name = f.name.lower()
        if Z is not None and z_str3 and z_str3.lower() in name:
            candidates.append(f)
            continue
        if sym is not None and sym.lower() in name:
            candidates.append(f)
            continue
        # fallback: candidate if contains 'mu_tau' or 'extracted'
        if "mu_tau" in name or "extracted" in name:
            candidates.append(f)
    # unique & prefer better matches
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None
    # prefer exact sym match
    for c in candidates:
        if sym and sym.lower() in c.name.lower():
            return c
    return candidates[0]

# ---------------------------
# main computation
def run(spectrum_csv, element_token, thickness_mm,
        edpl_folder, attenuation_csv,
        acceptance_radius_mm, distance_mm, kbeta_ratio,
        out_csv):
    # read input spectrum
    spec = pd.read_csv(spectrum_csv)
    if not {'Energy_KeV', 'Photons_per_s'}.issubset(spec.columns):
        raise ValueError("spectrum CSV must have columns: Energy_KeV, Photons_per_s")
    E_in = spec['Energy_KeV'].to_numpy()
    phi_in = spec['Photons_per_s'].to_numpy()
    dE = np.gradient(E_in)  # bin widths (keV)
    # canonical element token
    Z, sym = parse_element_token(element_token)
    if Z is None and not sym:
        raise ValueError("Cannot parse element token. Use Z (e.g. 42) or symbol (e.g. Mo).")

    # find EPDL extracted file
    edpl_file = find_edpl_extracted_file(edpl_folder, Z, sym)
    if edpl_file is None:
        raise FileNotFoundError(f"Could not find an EPDL-extracted CSV in {edpl_folder} matching element {element_token}. "
                                "Make sure you have a CSV with tauK_over_rho and mu_over_rho columns for this element.")
    df_edpl = pd.read_csv(edpl_file)
    if not {'E_keV', 'tauK_over_rho_cm2_per_g', 'mu_over_rho_cm2_per_g'}.issubset(df_edpl.columns):
        # accept alternative column names
        if 'tauK_over_rho' in df_edpl.columns:
            df_edpl = df_edpl.rename(columns={'tauK_over_rho': 'tauK_over_rho_cm2_per_g'})
        if 'mu_over_rho' in df_edpl.columns:
            df_edpl = df_edpl.rename(columns={'mu_over_rho': 'mu_over_rho_cm2_per_g'})
    # check again
    if not {'E_keV', 'tauK_over_rho_cm2_per_g', 'mu_over_rho_cm2_per_g'}.issubset(df_edpl.columns):
        raise ValueError(f"EDPL extracted CSV {edpl_file} missing one of required columns: E_keV, tauK_over_rho_cm2_per_g, mu_over_rho_cm2_per_g")

    # load attenuation_coefficients_long
    att = pd.read_csv(attenuation_csv)
    if not {'Energy_KeV', 'Element', 'Mu_over_rho', 'Mu_en_over_rho'}.issubset(att.columns):
        raise ValueError("attenuation_coefficients_long CSV must contain columns: Energy_KeV, Element (number), Mu_over_rho, Mu_en_over_rho")

    # pick attenuation rows for this element
    if Z is None:
        # try to map sym -> Z via small map
        symbol_to_z = {"Mo":42, "Pb":82, "Cu":29, "W":74, "Au":79, "Ag":47}
        Z = symbol_to_z.get(sym, None)
        if Z is None:
            raise ValueError("Element Z not recognized and not provided; please give Z or update the script mapping.")
    att_elem = att[att['Element'] == Z]
    if att_elem.empty:
        raise ValueError(f"No rows in {attenuation_csv} for element Z={Z}")

    # interpolate mu and mu_en onto E_in
    E_att = att_elem['Energy_KeV'].to_numpy()
    mu_over_rho_att = att_elem['Mu_over_rho'].to_numpy()
    mu_en_over_rho_att = att_elem['Mu_en_over_rho'].to_numpy()
    mu_tot = np.interp(E_in, E_att, mu_over_rho_att, left=mu_over_rho_att[0], right=mu_over_rho_att[-1])
    mu_en = np.interp(E_in, E_att, mu_en_over_rho_att, left=mu_en_over_rho_att[0], right=mu_en_over_rho_att[-1])

    # interpolate tauK from edpl file onto E_in
    tauK = np.interp(E_in, df_edpl['E_keV'].to_numpy(), df_edpl['tauK_over_rho_cm2_per_g'].to_numpy(), left=0.0, right=0.0)

    # get density
    dens = DENSITIES_G_CM3.get(sym, None)
    if dens is None:
        # try some guesses from Z (Mo,Pb known)
        if Z == 42: dens = 10.28
        elif Z == 82: dens = 11.34
        else:
            raise ValueError(f"No density for element '{sym}' in script mapping. Add it to DENSITIES_G_CM3.")

    # convert thickness to cm, compute rho*t (g/cm^2)
    t_cm = float(thickness_mm) / 10.0
    rho_t = dens * t_cm

    # absorption probability P_abs(E)
    P_abs = 1.0 - np.exp(-mu_tot * rho_t)

    # fraction of absorbed that make K vacancies
    frac_K = np.zeros_like(mu_tot)
    nonzero = mu_tot > 0
    frac_K[nonzero] = tauK[nonzero] / mu_tot[nonzero]

    # R_K(E) vacancies per second per keV
    R_K = phi_in * P_abs * frac_K

    # get omega_K and K line energies from a local file if present
    # try Elements_omega_k_and_kab.csv in working dir
    omega_file = Path('Elements_omega_k_and_kab.csv')
    if omega_file.exists():
        omega_df = pd.read_csv(omega_file)
        # find row by Z or Element name
        row = None
        if 'Z' in omega_df.columns:
            row = omega_df[omega_df['Z'] == Z]
        if (row is None or row.empty) and 'Element' in omega_df.columns:
            # try element name matching
            row = omega_df[omega_df['Element'].str.contains(sym, case=False, na=False)]
        if row is not None and not row.empty:
            row0 = row.iloc[0]
            omega_K = float(row0.get(list(row0.filter(regex='omega', axis=0))[0])) if any('omega' in c.lower() for c in row0.index) else None
            # fallback column names
            if 'omega_K\u200b' in row0.index:
                omega_K = float(row0['omega_K\u200b'])
            E_Kalpha = float(row0.get('E_Kalpha', np.nan))
            E_Kbeta = float(row0.get('E_Kbeta', np.nan))
        else:
            omega_K = None
            E_Kalpha = np.nan
            E_Kbeta = np.nan
    else:
        omega_K = None
        E_Kalpha = np.nan
        E_Kbeta = np.nan

    # fallback defaults if not found
    if omega_K is None:
        # default reasonable guesses
        if Z == 42: omega_K = 0.765
        elif Z == 82: omega_K = 0.96
        else: omega_K = 0.7
    if math.isnan(E_Kalpha) or math.isnan(E_Kbeta):
        # fallback energy guesses for Mo/Pb
        if Z == 42:
            E_Kalpha, E_Kbeta = 17.479, 19.608
        elif Z == 82:
            E_Kalpha, E_Kbeta = 74.969, 84.936
        else:
            E_Kalpha, E_Kbeta = None, None

    # produced K-line photons (all directions) per sec (integrate)
    Y_Kprod_perkeV = R_K * omega_K
    Y_K_total = np.trapz(Y_Kprod_perkeV, E_in)  # photons/s
    # split lines using Kbeta ratio
    BR_Kalpha = 1.0 / (1.0 + kbeta_ratio)
    BR_Kbeta = 1.0 - BR_Kalpha
    Y_Kalpha_prod = Y_K_total * BR_Kalpha
    Y_Kbeta_prod  = Y_K_total * BR_Kbeta

    # escape probabilities for fluorescent photon energies (depth-averaged)
    # get mu_over_rho at emission energies from attenuation table
    if E_Kalpha is not None:
        mu_Kalpha = float(np.interp(E_Kalpha, E_att if 'E_att' in locals() else E_att, mu_over_rho_att))
        mu_Kbeta  = float(np.interp(E_Kbeta , E_att if 'E_att' in locals() else E_att, mu_over_rho_att))
    else:
        # if no emission energies known, skip fluorescence lines
        mu_Kalpha = mu_Kbeta = None

    def escape_prob(mu_over_rho_val):
        if mu_over_rho_val is None or mu_over_rho_val <= 0:
            return 1.0
        denom = mu_over_rho_val * rho_t
        if denom == 0:
            return 1.0
        return (1.0 - math.exp(-denom)) / denom

    P_esc_alpha = escape_prob(mu_Kalpha) if mu_Kalpha is not None else 0.0
    P_esc_beta  = escape_prob(mu_Kbeta)  if mu_Kbeta is not None else 0.0

    # acceptance fraction into small cone along axis (isotropic fraction)
    theta = math.atan2(acceptance_radius_mm, distance_mm)
    frac_accept = (1.0 - math.cos(theta)) / 2.0

    # escaping photons/s within acceptance cone
    Phi_alpha_esc_accept = Y_Kalpha_prod * P_esc_alpha * frac_accept
    Phi_beta_esc_accept  = Y_Kbeta_prod  * P_esc_beta  * frac_accept

    # Now compute transmitted continuum (attenuated primary)
    # transmitted per keV = phi_in * exp(-mu_tot*rho_t)
    phi_trans = phi_in * np.exp(-mu_tot * rho_t)

    # add fluorescence as delta lines placed in nearest energy bin(s)
    out_photons_per_keV = phi_trans.copy()
    # find bins (nearest index)
    if E_Kalpha is not None:
        idx_a = int(np.argmin(np.abs(E_in - E_Kalpha)))
        # convert photons/s to photons per keV by dividing by bin width
        out_photons_per_keV[idx_a] += Phi_alpha_esc_accept / dE[idx_a]
    if E_Kbeta is not None:
        idx_b = int(np.argmin(np.abs(E_in - E_Kbeta)))
        out_photons_per_keV[idx_b] += Phi_beta_esc_accept / dE[idx_b]

    # Save output CSV matching input format
    out_df = pd.DataFrame({'Energy_KeV': E_in, 'Photons_per_s': out_photons_per_keV})
    out_df.to_csv(out_csv, index=False)

    # Print summary
    print("Filter element:", sym, "Z=", Z)
    print("Thickness (mm):", thickness_mm, "density (g/cm3):", dens, "rho*t (g/cm2):", rho_t)
    print("Total incident photons (sum spectrum):", float(np.sum(phi_in * dE)))
    print("Total transmitted photons (sum output continuum):", float(np.sum(out_photons_per_keV * dE)))
    print("Produced K photons (total, all directions):", float(Y_K_total))
    print("Escaping Kalpha within acceptance cone (photons/s):", float(Phi_alpha_esc_accept))
    print("Escaping Kbeta within acceptance cone (photons/s):", float(Phi_beta_esc_accept))
    print(f"Acceptance cone half-angle (deg): {math.degrees(theta):.3f}, fraction of isotropic emission in cone: {frac_accept:.6f}")
    print("Wrote output CSV:", out_csv)

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
