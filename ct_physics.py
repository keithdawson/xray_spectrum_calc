"""Primary-ray target contrast and an idealized energy-integrating detector.

The detector retains energy transferred locally at the first interaction. It
assumes charged-particle energy is deposited locally and secondary photons
escape. It is not a finite-screen fluorescence/Compton transport calculation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from apply_filter_and_fluorescence import (
    AIR_COMPOSITION, DENSITIES, SYM_TO_Z, get_mixture_attenuation,
)
from ct_materials import MATERIAL_DEFINITIONS
from spectrum_utils import energy_bin_widths, integrate_spectrum

COMPOUND_COMPOSITIONS = {
    material: definition[1] for material, definition in MATERIAL_DEFINITIONS.items()
}
_ATT_DF_CACHE = {}


def _attenuation_table():
    if 'att' not in _ATT_DF_CACHE:
        _ATT_DF_CACHE['att'] = pd.read_csv(
            Path(__file__).resolve().with_name('attenuation_coefficients_long.csv')
        )
    return _ATT_DF_CACHE['att']


def material_coefficients(material, energies):
    """Return (mu/rho, mu_en/rho, density) from the same NIST table as filters.

    Mass-fraction additivity of mu_en/rho is approximate (radiative charged-
    particle losses are not strictly additive). At CT energies those corrections
    are small; see NIST XrayMassCoef, section 3.
    """
    if material in MATERIAL_DEFINITIONS:
        density, composition = MATERIAL_DEFINITIONS[material]
    elif material == 'Air':
        density, composition = DENSITIES['Air'], AIR_COMPOSITION
    elif material in SYM_TO_Z:
        density, composition = DENSITIES[material], {SYM_TO_Z[material]: 1.0}
    else:
        raise ValueError(f'Unsupported CT material: {material!r}')
    mu, mu_en = get_mixture_attenuation(
        np.asarray(energies, dtype=float), composition, _attenuation_table()
    )
    return mu, mu_en, density


def attenuation_exponent(material, energies, thickness_mm):
    """Narrow-beam exponent mu/rho * density * thickness_cm."""
    if not np.isfinite(thickness_mm) or thickness_mm < 0:
        raise ValueError('Material thickness must be finite and nonnegative.')
    mu, _, density = material_coefficients(material, energies)
    return mu * density * thickness_mm * 0.1


def _muen_over_mu(material, energies):
    """First-collision local energy fraction, not a screen escape probability."""
    mu, mu_en, _ = material_coefficients(material, energies)
    return np.divide(mu_en, mu, out=np.zeros_like(mu), where=mu > 0)


def detector_response(energies, scint_mat, scint_thick):
    """Return interaction probability, mean and second-moment kernels.

    With p = 1-exp(-mu*t) and q = E*mu_en/mu, a Poisson incident bin
    contributes mean N*p*q and variance N*p*q**2. This treats deposited
    energy q as deterministic conditional on interaction. Event-to-event
    energy-loss fluctuations (Swank noise), optical gain noise and electronic
    noise increase variance. Secondary-photon reabsorption also changes the
    mean; neither effect can be inferred from mu_en alone.

    NIST: https://physics.nist.gov/PhysRefData/XrayMassCoef/chap3.html
    Energy-integrating statistics: https://pubmed.ncbi.nlm.nih.gov/17404467/
    """
    energies = np.asarray(energies, dtype=float)
    probability = -np.expm1(-attenuation_exponent(scint_mat, energies, scint_thick))
    response = energies * _muen_over_mu(scint_mat, energies)
    return probability, probability * response, probability * response**2


def validate_spectrum(energies, photons):
    energies = np.asarray(energies, dtype=float)
    photons = np.asarray(photons, dtype=float)
    energy_bin_widths(energies)
    if photons.shape != energies.shape or not np.all(np.isfinite(photons)) or np.any(photons < 0):
        raise ValueError('Photon density must be finite, nonnegative and match the energy grid.')
    return energies, photons


def calculate_contrast(spectrum_df, target_a_mat, target_a_thick, target_b_mat, target_b_thick, scint_mat, scint_thick):
    """Compare independent primary rays through A and B (thicknesses in mm).

    Signal is deposited keV per unit exposure; variance is keV^2 per unit
    exposure. For photon density per second these are signal/variance rates.
    Neither object scatter nor CT reconstruction is included.
    """
    energies, photons_in = validate_spectrum(
        spectrum_df['Energy_KeV'].values, spectrum_df['Photons_per_s'].values
    )
    probability, mean_kernel, variance_kernel = detector_response(energies, scint_mat, scint_thick)
    result = {}
    for label, material, thickness in (
        ('A', target_a_mat, target_a_thick), ('B', target_b_mat, target_b_thick)
    ):
        photons_out = photons_in * np.exp(-attenuation_exponent(material, energies, thickness))
        energy_deposited = photons_out * mean_kernel
        variance_density = photons_out * variance_kernel
        result[f'signal_{label}'] = float(integrate_spectrum(energies, energy_deposited))
        result[f'variance_{label}'] = float(integrate_spectrum(energies, variance_density))
        result[f'df_{label}'] = pd.DataFrame({
            'Energy_KeV': energies,
            'Photons_per_s': photons_out,
            # Historical column: interactions, not full-energy absorptions.
            'Absorbed_Photons_per_s': photons_out * probability,
            'Energy_Deposited_keV_per_s': energy_deposited,
            'Variance_keV2_per_s': variance_density,
        })
    difference = abs(result['signal_A'] - result['signal_B'])
    result['contrast'] = difference / result['signal_A'] if result['signal_A'] > 0 else 0.0
    variance = result['variance_A'] + result['variance_B']
    result['cnr_per_sqrt_exposure'] = difference / np.sqrt(variance) if variance > 0 else 0.0
    result['detector_model'] = 'first_collision_deterministic_response'
    return result
