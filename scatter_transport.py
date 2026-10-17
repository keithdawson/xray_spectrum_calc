"""Single-Compton-scatter transport through a normally illuminated plane slab.

The total incoherent coefficient is supplied by the caller (normally the
binding-corrected table). When ``atomic_number`` is supplied, the angular
distribution uses the Waller--Hartree / Hubbell et al. (1975) atomic incoherent
scattering function ``S(q, Z)`` weighting the unpolarized Klein--Nishina cross
section, suppressing small-angle forward incoherent scattering where bound
electrons cannot absorb infinitesimal momentum transfers. When ``atomic_number``
is ``None``, the free-electron Klein--Nishina angular distribution is used.
The Compton energy shift and the longer oblique escape path are included.
Doppler broadening, coherent scatter and multiple collisions within one slab are
omitted. The acceptance cone describes directions about an incident ray; it is
not a detector/pixel geometry calculation.

Equations and limits: Geant4 Physics Reference Manual, Compton and Livermore:
https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html
https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/livermore_compton.html
"""

from functools import lru_cache

import numpy as np

from fluorescence_k_data import INCOHERENT_SF, INCOHERENT_SF_Q


ELECTRON_REST_ENERGY_KEV = 510.99895
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13
THOMSON_CROSS_SECTION_CM2 = 8.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 / 3.0
HC_KEV_ANGSTROM = 12.3984193

_SF_Q_GRID = np.asarray(INCOHERENT_SF_Q, dtype=float)
_LOG_SF_Q_GRID = np.log(_SF_Q_GRID)


@lru_cache(maxsize=96)
def _element_sf_table(atomic_number):
    if (not isinstance(atomic_number, (int, np.integer))
            or isinstance(atomic_number, bool)
            or int(atomic_number) not in INCOHERENT_SF):
        raise ValueError(f"Unsupported atomic number for incoherent scattering: {atomic_number!r}")
    z = int(atomic_number)
    sf = np.asarray(INCOHERENT_SF[z], dtype=float)
    log_sf = np.log(sf)
    sf.setflags(write=False)
    log_sf.setflags(write=False)
    return z, sf, log_sf


def compton_momentum_transfer(energy_keV, cos_theta):
    """Hubbell momentum-transfer parameter q = (E / hc) * sin(theta / 2) in 1/Angstrom."""
    energy = np.asarray(energy_keV, dtype=float)
    cosine = np.asarray(cos_theta, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Photon energies must be finite and positive.")
    if np.any(~np.isfinite(cosine)) or np.any(np.abs(cosine) > 1):
        raise ValueError("Scattering-angle cosines must lie in [-1, 1].")
    sin_half = np.sqrt(np.maximum(0.0, (1.0 - cosine) / 2.0))
    return (energy / HC_KEV_ANGSTROM) * sin_half


def incoherent_scattering_function(atomic_number, q_inv_angstrom):
    """Evaluate Waller--Hartree incoherent scattering function S(q, Z) in [0, Z].

    Uses log-log interpolation over the tabulated [0.005, 50] 1/Angstrom grid,
    quadratic dipole scaling S(q, Z) ~ q^2 below 0.005 1/Angstrom so S(0, Z) = 0,
    and saturates at Z above 50 1/Angstrom.
    """
    z, sf_grid, log_sf_grid = _element_sf_table(atomic_number)
    q = np.asarray(q_inv_angstrom, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("Momentum transfer q must be finite and nonnegative.")
    scalar_input = q.ndim == 0
    q_arr = np.atleast_1d(q)
    result = np.empty_like(q_arr, dtype=float)
    q_min = _SF_Q_GRID[0]
    q_max = _SF_Q_GRID[-1]
    low = q_arr < q_min
    high = q_arr >= q_max
    mid = ~(low | high)
    if np.any(low):
        result[low] = sf_grid[0] * (q_arr[low] / q_min) ** 2
    if np.any(high):
        result[high] = float(z)
    if np.any(mid):
        result[mid] = np.minimum(
            np.exp(np.interp(np.log(q_arr[mid]), _LOG_SF_Q_GRID, log_sf_grid)),
            float(z),
        )
    return float(result[0]) if scalar_input else result.reshape(q.shape)


def klein_nishina_total_cross_section(energy_keV):
    """Unpolarized free-electron total cross section, cm^2/electron.

    A low-energy series avoids cancellation in the usual closed form and tends
    to the Thomson cross section. This is not an atomic binding correction.
    """
    energy = np.asarray(energy_keV, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Photon energies must be finite and positive.")
    alpha = energy / ELECTRON_REST_ENERGY_KEV
    low = alpha < 0.01
    result = np.empty_like(alpha)
    # Expansion of sigma_KN / sigma_Thomson through alpha^8.
    result[low] = THOMSON_CROSS_SECTION_CM2 * np.polynomial.polynomial.polyval(
        alpha[low], [1.0, -2.0, 26 / 5, -133 / 10, 1144 / 35,
                     -544 / 7, 3784 / 21, -6148 / 15, 151552 / 165])
    k = alpha[~low]
    log_term = np.log1p(2.0 * k)
    result[~low] = 2.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2 * (
        (1.0 + k) / k**2 * (2.0 * (1.0 + k) / (1.0 + 2.0 * k) - log_term / k)
        + log_term / (2.0 * k) - (1.0 + 3.0 * k) / (1.0 + 2.0 * k)**2)
    return result


def klein_nishina_angular_density(energy_keV, cos_theta):
    """Probability density per d(cos(theta)), with azimuth integrated out.

    Inputs follow NumPy broadcasting. Integral from -1 to 1 is one, so this
    function can normalize an independently supplied total incoherent rate.
    """
    energy = np.asarray(energy_keV, dtype=float)
    cosine = np.asarray(cos_theta, dtype=float)
    if np.any(~np.isfinite(cosine)) or np.any(np.abs(cosine) > 1):
        raise ValueError("Scattering-angle cosines must lie in [-1, 1].")
    ratio = 1.0 / (1.0 + energy / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
    shape = ratio**2 * (ratio + 1.0 / ratio - 1.0 + cosine**2)
    normalizer = klein_nishina_total_cross_section(energy) / (
        np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2)
    return shape / normalizer


@lru_cache(maxsize=8)
def _legendre_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _waller_hartree_normalizer(energy_keV, atomic_number, order=64):
    """Integral over cos(theta) in [-1, 1] of KN_shape(E, cos(theta)) * S(q, Z).

    Uses the substitution s = sin(theta / 2) in [0, 1], cos(theta) = 1 - 2*s^2,
    d(cos(theta)) = -4*s ds, which removes the square-root singularity in q(cos)
    at cos(theta) = 1.
    """
    energy = np.asarray(energy_keV, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Photon energies must be finite and positive.")
    nodes, weights = _legendre_rule(order)
    s = 0.5 * (nodes + 1.0)
    cosine = 1.0 - 2.0 * s**2
    jacobian_weights = 2.0 * s * weights
    e_flat = np.atleast_1d(energy).ravel()[:, None]
    ratio = 1.0 / (1.0 + e_flat / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
    shape = ratio**2 * (ratio + 1.0 / ratio - 1.0 + cosine**2)
    q = (e_flat / HC_KEV_ANGSTROM) * s
    sf = incoherent_scattering_function(atomic_number, q)
    integral = np.sum(shape * sf * jacobian_weights, axis=-1)
    return float(integral[0]) if energy.ndim == 0 else integral.reshape(energy.shape)


def incoherent_angular_density(energy_keV, cos_theta, atomic_number=None):
    """Normalized incoherent angular probability density per d(cos(theta)).

    If ``atomic_number`` is ``None``, returns the free-electron Klein--Nishina
    angular density. If ``atomic_number`` is an element Z (1..92), weights the
    Klein--Nishina differential cross section by the Waller--Hartree incoherent
    scattering function S(q, Z) and normalizes over cos(theta) in [-1, 1].
    """
    if atomic_number is None:
        return klein_nishina_angular_density(energy_keV, cos_theta)
    energy = np.asarray(energy_keV, dtype=float)
    cosine = np.asarray(cos_theta, dtype=float)
    if np.any(~np.isfinite(cosine)) or np.any(np.abs(cosine) > 1):
        raise ValueError("Scattering-angle cosines must lie in [-1, 1].")
    ratio = 1.0 / (1.0 + energy / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
    shape = ratio**2 * (ratio + 1.0 / ratio - 1.0 + cosine**2)
    q = compton_momentum_transfer(energy, cosine)
    sf = incoherent_scattering_function(atomic_number, q)
    normalizer = _waller_hartree_normalizer(energy, atomic_number)
    return (shape * sf) / normalizer


def _depth_integral(mu_in, mu_out_oblique, mass_thickness):
    """Integral exp(-mu_in*x - mu_out_oblique*(T-x)) dx, without cancellation."""
    delta = np.abs(mu_in - mu_out_oblique) * mass_thickness
    exprel = np.ones_like(delta)
    np.divide(-np.expm1(-delta), delta, out=exprel, where=delta != 0)
    return (mass_thickness * np.exp(-np.minimum(mu_in, mu_out_oblique)
                                    * mass_thickness) * exprel)


def _adaptive_scatter_counts(energy, incident_counts, mu_total, mu_incoherent,
                             mass_thickness, cos_min, attenuation_at_energy,
                             atomic_number=None):
    """Integrate separately between output energy centers to resolve line shapes.

    Within each interval the deposition weights are smooth. Four angles are
    sufficient for its short Compton energy range; wide angular intervals or
    forward-angle binding-suppression intervals use sixteen, and intervals
    touching a grazing ray use thirty-two. Bounded batches avoid allocating an
    energy-by-energy-by-angle dense tensor.
    """
    source = np.flatnonzero((incident_counts > 0) & (mu_incoherent > 0))
    scattered_min = energy[source] / (
        1.0 + energy[source] / ELECTRON_REST_ENERGY_KEV * (1.0 - cos_min))
    lower_start = np.searchsorted(energy, np.maximum(scattered_min, energy[0]), side="right") - 1
    interval_counts = source - lower_start
    source_for_interval = np.repeat(source, interval_counts)
    starts = np.cumsum(interval_counts) - interval_counts
    lower_bin = (np.repeat(lower_start, interval_counts) + np.arange(interval_counts.sum())
                 - np.repeat(starts, interval_counts))
    out_counts = np.zeros_like(energy)
    if lower_bin.size == 0:
        return out_counts

    incident_energy = energy[source_for_interval]
    cos_left = np.maximum(cos_min, 1.0 - ELECTRON_REST_ENERGY_KEV * (
        1.0 / energy[lower_bin] - 1.0 / incident_energy))
    cos_right = np.minimum(1.0, 1.0 - ELECTRON_REST_ENERGY_KEV * (
        1.0 / energy[lower_bin + 1] - 1.0 / incident_energy))
    span = cos_right - cos_left
    # Near grazing incidence the escape path varies on the scale of cos(theta);
    # near cos(theta) -> 1 atomic binding S(q, Z) varies rapidly with q.
    near_forward = (cos_right > 0.95) if atomic_number is not None else False
    orders = np.where((span > 0.03) | (span > 0.25 * cos_left) | near_forward, 16, 4)
    orders[(cos_left < 1e-12) | ((atomic_number is not None) & (cos_right > 1.0 - 1e-9))] = 32
    # The total-rate normalizer depends only on incident energy.
    if atomic_number is None:
        normalization = klein_nishina_total_cross_section(energy) / (
            np.pi * CLASSICAL_ELECTRON_RADIUS_CM**2)
    else:
        normalization = _waller_hartree_normalizer(energy, atomic_number)
    for order in (4, 16, 32):
        intervals = np.flatnonzero(orders == order)
        nodes, weights = _legendre_rule(order)
        for start in range(0, intervals.size, 8192):
            chunk = intervals[start:start + 8192]
            source_index = source_for_interval[chunk]
            lower = lower_bin[chunk]
            half_span = span[chunk, None] / 2.0
            cosine = cos_left[chunk, None] + (nodes + 1.0) * half_span
            incident = energy[source_index, None]
            ratio = 1.0 / (1.0 + incident / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
            outgoing = incident * ratio
            # Roundoff at an endpoint must not request extrapolation from a
            # strict attenuation interpolator.
            outgoing = np.clip(outgoing, energy[lower, None], energy[lower + 1, None])
            mu_out = np.broadcast_to(
                np.asarray(attenuation_at_energy(outgoing.ravel()), dtype=float), outgoing.size).reshape(outgoing.shape)
            if np.any(~np.isfinite(mu_out)) or np.any(mu_out < 0):
                raise ValueError("Outgoing attenuation must be finite and nonnegative.")
            transfer = _depth_integral(mu_total[source_index, None], mu_out / cosine, mass_thickness)
            shape = ratio**2 * (ratio + 1.0 / ratio - 1.0 + cosine**2)
            if atomic_number is not None:
                q = (incident / HC_KEV_ANGSTROM) * np.sqrt(np.maximum(0.0, (1.0 - cosine) / 2.0))
                shape = shape * incoherent_scattering_function(atomic_number, q)
            counts = (incident_counts[source_index, None] * mu_incoherent[source_index, None]
                      * transfer * shape / normalization[source_index, None] * weights * half_span)
            fraction = (outgoing - energy[lower, None]) / (energy[lower + 1] - energy[lower])[:, None]
            out_counts += np.bincount(lower, weights=np.sum(counts * (1.0 - fraction), axis=1),
                                      minlength=energy.size)
            out_counts += np.bincount(lower + 1, weights=np.sum(counts * fraction, axis=1),
                                      minlength=energy.size)
    return out_counts


def single_scatter_spectrum(energy_keV, phi_density, mu_total, mu_incoherent,
                            mass_thickness, beam_angle_deg, attenuation_at_energy,
                            *, bin_widths=None, angular_order=None,
                            atomic_number=None):
    """Return the accepted singly scattered photon-fluence density per keV.

    ``mu_total`` and ``mu_incoherent`` are cm^2/g arrays on ``energy_keV``;
    ``mass_thickness`` is g/cm^2. ``attenuation_at_energy`` must accept an array
    of scattered energies and return total mass attenuation in cm^2/g. Only
    energies within the original spectrum's center range are passed to it.

    ``beam_angle_deg`` is the full cone angle, from 0 to 180 degrees (the latter
    accepts the complete downstream hemisphere). Spectra represent midpoint
    bins with widths ``np.gradient(energy_keV)`` unless explicitly supplied.
    Counts are redistributed linearly between adjacent energy centers to
    preserve both photon count and the first energy moment. Photons shifted
    below the represented energy range leave the calculation, rather than being
    placed artificially into its first bin.

    The default integrates angular intervals between adjacent output energy
    centers, using 4--32 Gauss--Legendre nodes per interval. This resolves the
    continuous Compton distribution of an input line even at broad cone angles.
    Set ``angular_order`` to an integer for a fixed whole-cone quadrature rule
    (useful for convergence checks); too few angles can introduce false peaks.
    When ``atomic_number`` is supplied (Z=1..92), the angular distribution is
    weighted by the Waller--Hartree atomic incoherent scattering function
    ``S(q, Z)``, accounting for bound-electron suppression of small-angle
    forward incoherent scatter.
    """
    energy = np.asarray(energy_keV, dtype=float)
    if (energy.ndim != 1 or energy.size < 2 or np.any(~np.isfinite(energy))
            or np.any(energy <= 0) or np.any(np.diff(energy) <= 0)):
        raise ValueError("Energy grid must contain at least two increasing positive energies.")
    spectrum = np.asarray(phi_density, dtype=float)
    if spectrum.shape != energy.shape or np.any(~np.isfinite(spectrum)) or np.any(spectrum < 0):
        raise ValueError("Photon density must be finite, nonnegative, and match the energy grid.")
    mu_total = np.broadcast_to(np.asarray(mu_total, dtype=float), energy.shape)
    mu_incoherent = np.broadcast_to(np.asarray(mu_incoherent, dtype=float), energy.shape)
    if (np.any(~np.isfinite(mu_total)) or np.any(mu_total < 0)
            or np.any(~np.isfinite(mu_incoherent)) or np.any(mu_incoherent < 0)):
        raise ValueError("Attenuation coefficients must be finite and nonnegative.")
    if not np.isfinite(mass_thickness) or mass_thickness < 0:
        raise ValueError("Mass thickness must be finite and nonnegative.")
    if not np.isfinite(beam_angle_deg) or not 0 <= beam_angle_deg <= 180:
        raise ValueError("Full acceptance-cone angle must lie between 0 and 180 degrees.")
    if angular_order is not None and (
            not isinstance(angular_order, (int, np.integer)) or angular_order < 2):
        raise ValueError("Angular quadrature order must be None or an integer of at least two.")
    if atomic_number is not None:
        _element_sf_table(atomic_number)
    widths = np.gradient(energy) if bin_widths is None else np.asarray(bin_widths, dtype=float)
    if widths.shape != energy.shape or np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Energy-bin widths must be positive and match the energy grid.")
    active = (spectrum > 0) & (mu_incoherent > 0)
    if mass_thickness == 0 or beam_angle_deg == 0 or not np.any(active):
        return np.zeros_like(energy)

    cos_min = np.cos(np.deg2rad(beam_angle_deg / 2.0))
    if cos_min == 1.0:
        return np.zeros_like(energy)
    if angular_order is None:
        return _adaptive_scatter_counts(
            energy, spectrum * widths, mu_total, mu_incoherent, mass_thickness,
            cos_min, attenuation_at_energy, atomic_number=atomic_number) / widths

    nodes, weights = _legendre_rule(angular_order)
    half_span = (1.0 - cos_min) / 2.0
    cosine = cos_min + (nodes + 1.0) * half_span
    angular_weights = weights * half_span
    incident_energy = energy[active, None]
    ratio = 1.0 / (1.0 + incident_energy / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
    outgoing_energy = incident_energy * ratio
    represented = (outgoing_energy >= energy[0]) & (outgoing_energy <= energy[-1])
    if not np.any(represented):
        return np.zeros_like(energy)

    mu_out = np.zeros_like(outgoing_energy)
    mu_out[represented] = attenuation_at_energy(outgoing_energy[represented])
    if np.any(~np.isfinite(mu_out)) or np.any(mu_out < 0):
        raise ValueError("Outgoing attenuation must be finite and nonnegative.")
    transfer = _depth_integral(mu_total[active, None], mu_out / cosine, mass_thickness)
    angular_probability = incoherent_angular_density(
        incident_energy, cosine, atomic_number=atomic_number) * angular_weights
    incident_counts = spectrum[active] * widths[active]
    counts = (incident_counts[:, None] * mu_incoherent[active, None]
              * transfer * angular_probability)[represented]
    outgoing_energy = outgoing_energy[represented]

    lower = np.searchsorted(energy, outgoing_energy, side="right") - 1
    lower = np.clip(lower, 0, energy.size - 2)
    upper_fraction = (outgoing_energy - energy[lower]) / (energy[lower + 1] - energy[lower])
    out_counts = (np.bincount(lower, weights=counts * (1.0 - upper_fraction), minlength=energy.size)
                  + np.bincount(lower + 1, weights=counts * upper_fraction, minlength=energy.size))
    return out_counts / widths
