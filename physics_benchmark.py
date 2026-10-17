"""Reproducible secondary-radiation benchmark; run from the project folder."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spekpy

from apply_filter_and_fluorescence import process_single_layer


def benchmark():
    root = Path(__file__).resolve().parent
    attenuation = pd.read_csv(root / "attenuation_coefficients_long.csv")
    source = spekpy.Spek(kvp=100, th=12, dk=0.1)
    source.filter("Al", 1.0)
    energies, incident = source.get_spectrum()
    widths = np.gradient(energies)
    results = []
    for material, thickness in [("Cu", 0.05), ("Mo", 0.05), ("Ta", 0.05), ("W", 0.1)]:
        total, fluorescence, scatter, transmission = process_single_layer(
            energies, incident, material, thickness, attenuation,
            root / "pyne_edpl", {"beam_angle_deg": 30.0},
        )
        results.append({
            "material": material, "thickness_mm": thickness,
            "primary": float(np.sum(incident * transmission * widths)),
            "fluorescence": float(np.sum(fluorescence * widths)),
            "scatter": float(np.sum(scatter * widths)),
            "fluorescence_energy": float(np.sum(fluorescence * energies * widths)),
            "scatter_energy": float(np.sum(scatter * energies * widths)),
            "total": float(np.sum(total * widths)),
        })
    return results


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
