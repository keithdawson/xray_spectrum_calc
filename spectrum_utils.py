"""Numerics for spectra stored as densities at increasing energy-bin centres.

Interior bin boundaries are centre midpoints; outer half-widths use the
nearest spacing. Integrals therefore include full first/last bins.
"""
import numpy as np


def energy_bin_widths(energies):
    energies = np.asarray(energies, dtype=float)
    if (energies.ndim != 1 or energies.size < 2
            or not np.all(np.isfinite(energies)) or np.any(energies <= 0)
            or np.any(np.diff(energies) <= 0)):
        raise ValueError("Energy bins must have at least two finite, positive, strictly increasing centres.")
    return np.gradient(energies)


def integrate_spectrum(energies, density, axis=-1):
    widths = energy_bin_widths(energies)
    density = np.asarray(density, dtype=float)
    if density.shape[axis] != widths.size:
        raise ValueError("Spectrum length does not match its energy grid.")
    return np.sum(np.moveaxis(density, axis, -1) * widths, axis=-1)


def interpolate_attenuation(energies, table_energies, coefficients):
    """Power-law interpolation retaining the table's closely spaced edges.

    No constant extrapolation: coefficients outside the supported range
    cannot be inferred reliably. Generated tables encode the edge sides
    as distinct, closely spaced energies.
    """
    query = np.asarray(energies, dtype=float)
    table = np.asarray(table_energies, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    if (table.ndim != 1 or table.size < 2 or values.shape != table.shape
            or np.any(np.diff(table) <= 0) or np.any(table <= 0)
            or not np.all(np.isfinite(table)) or np.any(values <= 0)
            or not np.all(np.isfinite(values))):
        raise ValueError("Attenuation data must be positive, finite and ordered by unique energy.")
    if (not np.all(np.isfinite(query)) or np.any(query < table[0])
            or np.any(query > table[-1])):
        raise ValueError(f"Photon energies must lie within attenuation data range {table[0]:g}–{table[-1]:g} keV.")
    return np.exp(np.interp(np.log(query), np.log(table), np.log(values)))


def add_line_density(output, energies, line_energy, photons, widths=None):
    """Deposit a line between two centres, conserving photons and energy.

    Lines outside the represented centre range are omitted, never moved
    to an unrelated boundary energy. Use a wider/finer grid to resolve them.
    """
    energies = np.asarray(energies, dtype=float)
    if line_energy < energies[0] or line_energy > energies[-1] or photons == 0:
        return
    widths = energy_bin_widths(energies) if widths is None else widths
    high = min(int(np.searchsorted(energies, line_energy, side="right")), len(energies) - 1)
    low = max(0, high - 1)
    fraction = (line_energy - energies[low]) / (energies[high] - energies[low])
    output[low] += photons * (1.0 - fraction) / widths[low]
    output[high] += photons * fraction / widths[high]
