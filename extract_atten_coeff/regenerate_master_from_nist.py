"""
Regenerates attenuation_coefficients_long.csv directly from the local NIST
XrayMassCoef HTML mirror (physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab).

Replaces the old two-step pipeline (convert_html_tables_to_csv.py +
generate_master_attenuation.py), fixing two accuracy problems:

  1. Log-log interpolation between NIST anchor energies. mu/rho follows a
     power law between anchors, so the old linear interpolation overestimated
     it by up to ~10% in the sparsely tabulated 60-300 keV region.
  2. Absorption edges preserved. NIST duplicates the edge energy with the
     below-edge and above-edge values; these are kept as a pair of points
     separated by a tiny epsilon so downstream linear interpolation
     reproduces a sharp step at the true edge energy instead of a ramp
     smeared across a full 1 keV grid cell.

The output grid per element is the union of a 1 keV comb (1-500 keV) and the
original NIST anchor/edge energies, so tabulated NIST values appear exactly.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

MIRROR = Path(__file__).resolve().parent.parent / 'physics.nist.gov' / 'PhysRefData' / 'XrayMassCoef' / 'ElemTab'
OUT_PATH = Path(__file__).resolve().parent.parent / 'attenuation_coefficients_long.csv'

MAX_KEV = 500.0

# Matches NIST <PRE> data rows: optional shell label, then E [MeV], mu/rho, mu_en/rho
DATA_LINE = re.compile(
    r'^\s*(?:[A-Za-z0-9]+\s+)?([0-9.]+E[+-][0-9]+)\s+([0-9.]+E[+-][0-9]+)\s+([0-9.]+E[+-][0-9]+)\s*$'
)


def parse_nist_html(path):
    """Returns (E_MeV, mu_over_rho, mu_en_over_rho) arrays with edge
    duplicates bumped by a tiny epsilon so energies strictly increase."""
    content = path.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'<PRE>(.*?)</PRE>', content, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError(f'No <PRE> table in {path}')

    rows = []
    last_energy = None
    for line in m.group(1).split('\n'):
        dm = DATA_LINE.match(line)
        if not dm:
            continue
        e, mu, mu_en = (float(g) for g in dm.groups())
        if last_energy is not None and e == last_energy:
            # Above-edge row: nudge so the step survives as two points
            e_out = e * 1.00001
        else:
            e_out = e
        last_energy = float(dm.group(1))
        rows.append((e_out, mu, mu_en))

    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def loglog_interp(x_new, x, y):
    return np.exp(np.interp(np.log(x_new), np.log(x), np.log(y)))


def main():
    html_files = sorted(MIRROR.glob('z[0-9][0-9].html'))
    if not html_files:
        raise SystemExit(f'No zNN.html files found in {MIRROR}')

    comb_kev = np.arange(1.0, MAX_KEV + 1.0)
    all_data = []

    for path in html_files:
        z = int(path.stem[1:])
        e_mev, mu, mu_en = parse_nist_html(path)
        e_kev = e_mev * 1000.0

        # Keep anchors up to the first point at/above MAX_KEV so the comb
        # can be interpolated (not extrapolated) at the top of the range.
        top = np.searchsorted(e_kev, MAX_KEV, side='right')
        if top < len(e_kev):
            top += 1
        e_kev, mu, mu_en = e_kev[:top], mu[:top], mu_en[:top]

        anchors = e_kev[(e_kev >= comb_kev[0]) & (e_kev <= MAX_KEV)]
        grid = np.union1d(comb_kev, anchors)

        grid_mu = loglog_interp(grid, e_kev, mu)
        grid_mu_en = loglog_interp(grid, e_kev, mu_en)

        # Snap grid points that coincide with anchors to the exact NIST
        # values (union1d/log round-trips can leave tiny drift).
        idx = np.searchsorted(e_kev, grid)
        idx = np.clip(idx, 0, len(e_kev) - 1)
        exact = np.isclose(e_kev[idx], grid, rtol=1e-12, atol=0)
        grid_mu[exact] = mu[idx[exact]]
        grid_mu_en[exact] = mu_en[idx[exact]]

        all_data.append(pd.DataFrame({
            'Energy_MeV': grid / 1000.0,
            'Energy_KeV': grid,
            'Element': z,
            'Mu_over_rho': grid_mu,
            'Mu_en_over_rho': grid_mu_en,
        }))

    master = pd.concat(all_data, ignore_index=True)
    master = master.sort_values(by=['Element', 'Energy_KeV'], kind='stable')
    master.to_csv(OUT_PATH, index=False, float_format='%.6g')
    n_elem = master['Element'].nunique()
    print(f'Wrote {OUT_PATH} ({len(master)} rows, {n_elem} elements)')


if __name__ == '__main__':
    main()
