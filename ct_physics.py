import numpy as np
import pandas as pd
import spekpy

from apply_filter_and_fluorescence import SYM_TO_Z

# Mass-fraction compositions for compound scintillators (must stay in sync
# with the definitions in ct_materials.py)
COMPOUND_COMPOSITIONS = {
    'Gadox': {64: 0.831, 8: 0.085, 16: 0.085},
    'CsI': {55: 0.512, 53: 0.488},
}

_ATT_DF_CACHE = {}


def _muen_over_mu(material, energies):
    """Ratio of mass energy-absorption to mass attenuation coefficient for
    a material, from the NIST element table. Fraction of an interacting
    photon's energy that is locally deposited (the rest leaves as scattered
    or fluorescence photons)."""
    if 'att' not in _ATT_DF_CACHE:
        _ATT_DF_CACHE['att'] = pd.read_csv('attenuation_coefficients_long.csv')
    att_df = _ATT_DF_CACHE['att']

    comp = COMPOUND_COMPOSITIONS.get(material)
    if comp is None:
        z = SYM_TO_Z.get(material)
        if z is None:
            return np.ones_like(energies, dtype=float)
        comp = {z: 1.0}

    mu = np.zeros_like(energies, dtype=float)
    mu_en = np.zeros_like(energies, dtype=float)
    for z, w in comp.items():
        el = att_df[att_df['Element'] == z]
        mu += w * np.interp(energies, el['Energy_KeV'], el['Mu_over_rho'])
        mu_en += w * np.interp(energies, el['Energy_KeV'], el['Mu_en_over_rho'])

    return np.where(mu > 0, mu_en / mu, 1.0)


def calculate_contrast(spectrum_df, target_a_mat, target_a_thick, target_b_mat, target_b_thick, scint_mat, scint_thick):
    """
    Calculates the CT contrast between Target A and Target B as seen by a given scintillator.
    
    :param spectrum_df: Pandas DataFrame with columns 'Energy_KeV' and 'Photons_per_s'
    :param target_a_mat: string, name of material A
    :param target_a_thick: float, thickness of material A in mm
    :param target_b_mat: string, name of material B
    :param target_b_thick: float, thickness of material B in mm
    :param scint_mat: string, name of scintillator material
    :param scint_thick: float, thickness of scintillator in mm
    :return: dict containing signals and contrast metrics, and DataFrames of the absorbed spectra
    """
    # Load SpekPy mu data source (pene_mu.dat contains the comprehensive NIST data)
    mu_data = spekpy.SpekTools.MuData('pene_mu.dat')
    
    energies = spectrum_df['Energy_KeV'].values
    photons_in = spectrum_df['Photons_per_s'].values
    
    # 1. Transmission through Targets
    # Mu_t is mu(E) * thickness
    mu_t_A = mu_data.get_mu_t(target_a_mat, energies, target_a_thick)
    mu_t_B = mu_data.get_mu_t(target_b_mat, energies, target_b_thick)
    
    photons_out_A = photons_in * np.exp(-mu_t_A)
    photons_out_B = photons_in * np.exp(-mu_t_B)
    
    # 2. Scintillator Absorption
    mu_t_scint = mu_data.get_mu_t(scint_mat, energies, scint_thick)
    absorption_prob = 1.0 - np.exp(-mu_t_scint)
    
    photons_absorbed_A = photons_out_A * absorption_prob
    photons_absorbed_B = photons_out_B * absorption_prob
    
    # 3. Energy Deposition (Signal)
    # Deposited energy per interacting photon is weighted by mu_en/mu:
    # Compton-scattered and fluorescence photons carry part of the energy
    # out of the scintillator (e.g. Gd K-escape at ~43 keV), so counting
    # full E per interaction would overestimate the signal at high energy.
    deposit_frac = _muen_over_mu(scint_mat, energies)
    energy_deposited_A = photons_absorbed_A * energies * deposit_frac
    energy_deposited_B = photons_absorbed_B * energies * deposit_frac
    # Integrate (trapz is deprecated in numpy 2.0, renamed to trapezoid)
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    
    total_signal_A = trapz(energy_deposited_A, energies)
    total_signal_B = trapz(energy_deposited_B, energies)
    
    # Weber Contrast (Assuming A is Background)
    if total_signal_A > 0:
        contrast = abs(total_signal_A - total_signal_B) / total_signal_A
    else:
        contrast = 0.0
        
    # Formatting output for visualization
    df_A = pd.DataFrame({
        'Energy_KeV': energies,
        'Photons_per_s': photons_out_A,
        'Absorbed_Photons_per_s': photons_absorbed_A,
        'Energy_Deposited_keV_per_s': energy_deposited_A
    })
    
    df_B = pd.DataFrame({
        'Energy_KeV': energies,
        'Photons_per_s': photons_out_B,
        'Absorbed_Photons_per_s': photons_absorbed_B,
        'Energy_Deposited_keV_per_s': energy_deposited_B
    })
    
    return {
        'signal_A': total_signal_A,
        'signal_B': total_signal_B,
        'contrast': contrast,
        'df_A': df_A,
        'df_B': df_B
    }
