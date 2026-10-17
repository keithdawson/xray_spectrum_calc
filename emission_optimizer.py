"""Optimize primary-beam energy fluence above a configurable threshold.

The historical module/function names are retained for callers. The high band
contains bremsstrahlung as well as characteristic lines; this is not a line-
purity objective. Secondary fluorescence and scatter are not included here.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spekpy as sp

from spectrum_utils import interpolate_attenuation

DATA_DIR = Path(__file__).resolve().parent
CACHE_VERSION = 2


def _best_per_material(df):
    """Return the best-scoring thickness of each material, ranked by score."""
    best_idx = df.groupby('Filter_Material')['Score'].idxmax()
    return df.loc[best_idx].sort_values(by='Score', ascending=False).reset_index(drop=True)


def _band_energy_weights(energies, dk, threshold_kev):
    """Integrate E dE over each uniform SpekPy bin on both sides of a cut.

    Fluence is a density per keV at bin centers. Splitting the bin containing
    the threshold avoids losing an interval, including when dk does not
    divide the threshold. Low + high equals E_center * dk for every bin.
    """
    energies = np.asarray(energies, dtype=float)
    left = energies - 0.5 * dk
    right = energies + 0.5 * dk
    cut = np.clip(threshold_kev, left, right)
    low = 0.5 * (cut - left) * (cut + left)
    high = 0.5 * (right - cut) * (right + cut)
    return low, high


def _cache_path(settings, attenuation_path, materials, densities):
    """Include all spectral settings and data used by this approximation."""
    metadata = {
        'version': CACHE_VERSION,
        'spekpy': getattr(sp, '__version__', 'unknown'),
        'settings': settings,
        'attenuation_sha256': hashlib.sha256(attenuation_path.read_bytes()).hexdigest(),
        'materials': [(sym, densities[sym]) for sym in materials],
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    return DATA_DIR / f'emission_opt_results_v{CACHE_VERSION}_{digest}.csv'


def run_emission_optimizer(kvp=220, max_thickness_mm=2.0, th=12, targ='W', dk=0.1,
                           *, threshold_kev=57.0):
    """Find a primary-beam hardening filter for each practical material.

    Maximize (removed low-band energy fluence %) * (retained high-band
    energy fluence %)^2. The high band begins at threshold_kev (57 by
    default); it includes the entire high-energy continuum. Results are
    screening estimates before filter-generated fluorescence/scatter and
    should be evaluated with the full stack model before selecting a filter.
    """
    from apply_filter_and_fluorescence import (
        DENSITIES, PRACTICAL_FILTER_MATERIALS, SYM_TO_Z,
    )

    kvp, max_thickness_mm, th, dk, threshold_kev = map(
        float, (kvp, max_thickness_mm, th, dk, threshold_kev))
    if not np.all(np.isfinite([kvp, max_thickness_mm, th, dk, threshold_kev])):
        raise ValueError('Spectrum and thickness settings must be finite.')
    if dk <= 0 or max_thickness_mm < 0.01:
        raise ValueError('Energy bin width must be positive and maximum thickness at least 0.01 mm.')
    if not 0 < threshold_kev < kvp:
        raise ValueError('The energy threshold must be positive and below the tube voltage.')

    attenuation_path = DATA_DIR / 'attenuation_coefficients_long.csv'
    settings = dict(kvp=kvp, max_thickness_mm=max_thickness_mm, th=th,
                    targ=targ, dk=dk, threshold_kev=threshold_kev)
    cache_file = _cache_path(settings, attenuation_path, PRACTICAL_FILTER_MATERIALS, DENSITIES)
    if cache_file.exists():
        return _best_per_material(pd.read_csv(cache_file))

    print(f'Running high-energy fluence optimizer for {kvp:g} kVp...')
    s = sp.Spek(kvp=kvp, th=th, targ=targ, dk=dk)
    # edges=True creates duplicated staircase coordinates for plotting,
    # not the energy centers required for attenuation calculations.
    energies, fluxes = s.get_spectrum(edges=False)
    low_weights, high_weights = _band_energy_weights(energies, dk, threshold_kev)
    low_power_bins = fluxes * low_weights
    high_power_bins = fluxes * high_weights
    power_base_high = np.sum(high_power_bins)
    power_base_low = np.sum(low_power_bins)
    if power_base_low <= 0 or power_base_high <= 0:
        raise ValueError('The source spectrum must have nonzero energy fluence in both bands.')

    att_df = pd.read_csv(attenuation_path)
    num_steps = int(np.floor(max_thickness_mm / 0.01 + 1e-10))
    thicknesses_mm = np.arange(1, num_steps + 1, dtype=float) * 0.01
    if not np.isclose(thicknesses_mm[-1], max_thickness_mm, rtol=0, atol=1e-10):
        thicknesses_mm = np.append(thicknesses_mm, max_thickness_mm)
    results = []

    for sym in PRACTICAL_FILTER_MATERIALS:
        z = SYM_TO_Z[sym]
        elem_att = att_df[att_df['Element'] == z]
        if elem_att.empty:
            raise ValueError(f'Missing attenuation data for {sym} (Z={z}).')
        mu_tot = interpolate_attenuation(
            energies, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
        rho_t = (DENSITIES[sym] * thicknesses_mm / 10.0).reshape(-1, 1)
        trans = np.exp(-rho_t * mu_tot.reshape(1, -1))
        power_filt_high = np.sum(trans * high_power_bins, axis=1)
        power_filt_low = np.sum(trans * low_power_bins, axis=1)
        pct_kept_high = power_filt_high / power_base_high * 100.0
        pct_removed_low = (1.0 - power_filt_low / power_base_low) * 100.0
        score = pct_removed_low * pct_kept_high ** 2

        for i, thickness in enumerate(thicknesses_mm):
            results.append({
                'Filter_Material': sym,
                'Thickness_mm': round(float(thickness), 6),
                'Removed_Low_Pct': pct_removed_low[i],
                'Kept_High_Pct': pct_kept_high[i],
                'Score': score[i],
            })

    df_res = pd.DataFrame(results).sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_res.to_csv(cache_file, index=False)
    return _best_per_material(df_res)
