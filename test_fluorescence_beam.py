"""Optional visual diagnostic using the production physics engine."""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import spekpy

from apply_filter_and_fluorescence import process_single_layer


def main():
    root = Path(__file__).resolve().parent
    source = spekpy.Spek(kvp=100, th=12, dk=0.1)
    source.filter('Al', 1.)
    energy, incident = source.get_spectrum()
    attenuation = pd.read_csv(root / 'attenuation_coefficients_long.csv')
    total, fluorescence, _, transmission = process_single_layer(
        energy, incident, 'Mo', .05, attenuation, root / 'pyne_edpl',
        {'beam_angle_deg': 180.}, calc_scatter=False)
    paths = []
    for name, density in [('primary', incident * transmission), ('fluorescence', fluorescence), ('total', total)]:
        path = root / f'test_beam_{name}.csv'
        pd.DataFrame({'Energy_KeV': energy, 'Photons_per_s': density,
                      'Normalization': 'photons/cm2/s/keV'}).to_csv(path, index=False)
        paths.append(str(path))
    subprocess.run([sys.executable, str(root / 'plot_csv.py'), *paths], check=True)


if __name__ == '__main__':
    main()
