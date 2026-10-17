"""Analytic and conservation checks for single-scatter slab transport."""

import unittest

import numpy as np

from scatter_transport import (
    ELECTRON_REST_ENERGY_KEV,
    THOMSON_CROSS_SECTION_CM2,
    coherent_angular_density,
    coherent_form_factor,
    coherent_scatter_spectrum,
    incoherent_angular_density,
    incoherent_scattering_function,
    klein_nishina_angular_density,
    klein_nishina_total_cross_section,
    single_scatter_spectrum,
)


class TestScatterTransport(unittest.TestCase):
    def test_thomson_limit_and_cross_section_normalization(self):
        self.assertAlmostEqual(
            float(klein_nishina_total_cross_section(1e-8)) / THOMSON_CROSS_SECTION_CM2,
            1.0, places=9)
        cosines, weights = np.polynomial.legendre.leggauss(160)
        energies = np.array([1e-8, 0.1, 5.10, 5.12, 50.0, 225.0, 1000.0])
        probability = klein_nishina_angular_density(energies[:, None], cosines)
        np.testing.assert_allclose(probability @ weights, np.ones(energies.size), rtol=2e-10)
        self.assertTrue(np.all(np.diff(klein_nishina_total_cross_section(energies)) < 0))

    def test_thin_thomson_cone_probability(self):
        # Analytic unpolarized Thomson cone fraction: (4 - 3c - c^3) / 8.
        energy = np.array([0.00001, 0.00002, 0.00003])
        widths = np.gradient(energy)
        spectrum = np.array([0.0, 1.0 / widths[1], 0.0])
        thickness = 1e-8
        cosine = np.cos(np.deg2rad(30.0))
        cone_fraction = (4.0 - 3.0 * cosine - cosine**3) / 8.0
        result = single_scatter_spectrum(
            energy, spectrum, 1.0, 0.4, thickness, 60.0, lambda e: np.ones_like(e))
        self.assertAlmostEqual(np.sum(result * widths) / (0.4 * thickness), cone_fraction, places=7)

    def test_nonuniform_grid_preserves_count_and_compton_energy_moment(self):
        energy = np.array([1.0, 30.0, 50.0, 70.0, 100.0, 150.0, 225.0])
        widths = np.gradient(energy)
        spectrum = np.zeros_like(energy)
        spectrum[4] = 1.0 / widths[4]
        total_mu, scatter_mu, thickness = 0.3, 0.2, 0.4
        result = single_scatter_spectrum(
            energy, spectrum, total_mu, scatter_mu, thickness, 120.0,
            lambda e: np.zeros_like(e))

        # Independent dense angular integration, with analytically transparent
        # escape. Normalization of the supplied total coefficient is retained.
        cosine = np.linspace(0.5, 1.0, 100001)
        density = klein_nishina_angular_density(100.0, cosine)
        outgoing_energy = 100.0 / (1.0 + 100.0 / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
        collision_probability = scatter_mu * -np.expm1(-total_mu * thickness) / total_mu
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        expected_count = collision_probability * integrate(density, cosine)
        expected_energy = collision_probability * integrate(density * outgoing_energy, cosine)
        self.assertAlmostEqual(np.sum(result * widths) / expected_count, 1.0, places=9)
        self.assertAlmostEqual(np.sum(result * widths * energy) / expected_energy, 1.0, places=9)
        self.assertLess(np.sum(result * widths * energy) / np.sum(result * widths), 100.0)
        self.assertEqual(np.sum(result[energy > 100.0]), 0.0)

    def test_oblique_escape_matches_independent_depth_angle_integral(self):
        energy = np.linspace(1.0, 225.0, 225)
        spectrum = np.zeros_like(energy)
        spectrum[99] = 1.0
        total_mu, scatter_mu, thickness = 0.7, 0.2, 0.9
        outgoing_mu = 1.2
        result = single_scatter_spectrum(
            energy, spectrum, total_mu, scatter_mu, thickness, 180.0,
            lambda e: np.full_like(e, outgoing_mu), angular_order=64)
        # Numerical integration over both generation depth and exit angle;
        # no use of the production closed-form depth transfer.
        nodes, weights = np.polynomial.legendre.leggauss(256)
        cosine, angular_weights = (nodes + 1.0) / 2.0, weights / 2.0
        depth, depth_weights = (nodes + 1.0) * thickness / 2.0, weights * thickness / 2.0
        survival = np.exp(-total_mu * depth[:, None]
                          - outgoing_mu * (thickness - depth[:, None]) / cosine)
        expected = scatter_mu * np.sum(
            (depth_weights @ survival) * klein_nishina_angular_density(100.0, cosine) * angular_weights)
        self.assertAlmostEqual(np.sum(result) / expected, 1.0, places=7)
        # Treating all outgoing rays as normal overestimates escape.
        normal_escape = (np.exp(-total_mu * thickness) - np.exp(-outgoing_mu * thickness)) / (
            outgoing_mu - total_mu)
        cone_probability = np.sum(klein_nishina_angular_density(100.0, cosine) * angular_weights)
        self.assertLess(np.sum(result), scatter_mu * normal_escape * cone_probability)

    def test_photons_cannot_exceed_incoherent_first_interactions(self):
        energy = np.linspace(1.0, 225.0, 2241)
        widths = np.gradient(energy)
        spectrum = np.exp(-((energy - 90.0) / 35.0)**2)
        total_mu = 0.4 + 20.0 / energy
        incoherent_mu = np.full_like(energy, 0.2)
        for thickness in [1e-8, 0.1, 2.0, 1000.0]:
            with self.subTest(thickness=thickness):
                scatter = single_scatter_spectrum(
                    energy, spectrum, total_mu, incoherent_mu, thickness, 180.0,
                    lambda e: 0.4 + 20.0 / e)
                interactions = spectrum * incoherent_mu / total_mu * -np.expm1(-total_mu * thickness)
                self.assertTrue(np.all(np.isfinite(scatter)))
                self.assertTrue(np.all(scatter >= 0))
                self.assertLessEqual(np.sum(scatter * widths), np.sum(interactions * widths))
                self.assertLessEqual(np.sum(scatter * widths * energy),
                                     np.sum(interactions * widths * energy))

    def test_adaptive_quadrature_resolves_broad_cone_monochromatic_scatter(self):
        energy = np.linspace(1.0, 225.0, 2241)
        widths = np.gradient(energy)
        # Independent dense midpoint angular integration resolves the complete
        # continuous line shape. A 16-angle whole-cone rule has >50% L1 error
        # for the 60-keV line at 120 degrees, despite accurate total counts.
        for line, angle in [(60.0, 120.0), (60.0, 180.0), (200.0, 180.0)]:
            with self.subTest(line=line, angle=angle):
                spectrum = np.zeros_like(energy)
                source_bin = np.argmin(np.abs(energy - line))
                spectrum[source_bin] = 1.0 / widths[source_bin]
                actual = single_scatter_spectrum(
                    energy, spectrum, 0.3, 0.2, 0.1, angle, lambda e: np.full_like(e, 0.3))
                cos_min = np.cos(np.deg2rad(angle / 2.0))
                dc = (1.0 - cos_min) / 200000
                cosine = cos_min + (np.arange(200000) + 0.5) * dc
                ratio = 1.0 / (1.0 + line / ELECTRON_REST_ENERGY_KEV * (1.0 - cosine))
                outgoing = line * ratio
                escape = (np.exp(-0.03) - np.exp(-0.03 / cosine)) / (0.3 / cosine - 0.3)
                counts = 0.2 * escape * klein_nishina_angular_density(line, cosine) * dc
                lower = np.searchsorted(energy, outgoing, side="right") - 1
                fraction = (outgoing - energy[lower]) / (energy[lower + 1] - energy[lower])
                reference = (np.bincount(lower, weights=counts * (1.0 - fraction), minlength=energy.size)
                             + np.bincount(lower + 1, weights=counts * fraction, minlength=energy.size))
                actual_counts = actual * widths
                self.assertLess(np.sum(np.abs(actual_counts - reference)) / np.sum(reference), 1e-6)
                self.assertAlmostEqual(np.sum(actual_counts) / np.sum(reference), 1.0, places=7)
                self.assertAlmostEqual(np.sum(actual_counts * energy) / np.sum(reference * energy),
                                       1.0, places=7)

    def test_out_of_grid_photons_are_lost_without_extrapolating_attenuation(self):
        energy = np.array([100.0, 101.0])
        spectrum = np.array([1.0, 0.0])

        def forbidden_callback(_):
            self.fail("Attenuation should not be requested for unrepresented photon energies.")

        result = single_scatter_spectrum(energy, spectrum, 1.0, 0.2, 0.1, 180.0, forbidden_callback)
        np.testing.assert_array_equal(result, np.zeros_like(energy))

    def test_zero_thickness_zero_cone_and_zero_flux(self):
        energy = np.array([1.0, 50.0, 100.0])
        for thickness, angle, spectrum in [(0.0, 60.0, np.ones(3)),
                                           (1.0, 0.0, np.ones(3)),
                                           (1.0, 60.0, np.zeros(3))]:
            result = single_scatter_spectrum(
                energy, spectrum, 1.0, 0.2, thickness, angle, lambda e: np.ones_like(e))
            np.testing.assert_array_equal(result, np.zeros_like(energy))

    def test_invalid_inputs_are_rejected(self):
        cases = [([100.0, 50.0], [1.0, 1.0], 0.1, 30.0),
                 ([50.0, 100.0], [-1.0, 1.0], 0.1, 30.0),
                 ([50.0, 100.0], [1.0, 1.0], -0.1, 30.0),
                 ([50.0, 100.0], [1.0, 1.0], 0.1, 181.0)]
        for energy, spectrum, thickness, angle in cases:
            with self.subTest(case=(energy, spectrum, thickness, angle)):
                with self.assertRaises(ValueError):
                    single_scatter_spectrum(energy, spectrum, 1.0, 0.2, thickness, angle,
                                            lambda e: np.ones_like(e))
        with self.assertRaises(ValueError):
            single_scatter_spectrum([50.0, 100.0], [1.0, 1.0], 1.0, 0.2, 0.1, 30.0,
                                    lambda e: np.ones_like(e), atomic_number=0)

    def test_waller_hartree_binding_suppresses_forward_scatter_and_normalizes(self):
        for z in (13, 29, 42, 74, 82):
            self.assertEqual(incoherent_scattering_function(z, 0.0), 0.0)
            self.assertAlmostEqual(incoherent_scattering_function(z, 60.0), float(z))
            self.assertTrue(np.all(np.diff(
                incoherent_scattering_function(z, np.geomspace(1e-4, 50.0, 50))) >= 0))

        # Integration in s = sin(theta/2) verifies unit normalization over [-1, 1].
        nodes, weights = np.polynomial.legendre.leggauss(128)
        s = 0.5 * (nodes + 1.0)
        cosines = 1.0 - 2.0 * s**2
        jacobian_weights = 2.0 * s * weights
        energies = np.array([10.0, 30.0, 60.0, 100.0, 225.0])
        for z in (13, 29, 42, 74):
            density = incoherent_angular_density(energies[:, None], cosines, atomic_number=z)
            np.testing.assert_allclose(density @ jacobian_weights, np.ones(energies.size), rtol=2e-5)
            np.testing.assert_array_equal(
                incoherent_angular_density(energies, 1.0, atomic_number=z), np.zeros(energies.size))

        # Bound-electron suppression reduces 30-deg forward-cone scatter vs free-electron KN,
        # while adaptive quadrature matches a 200,000-point reference integral.
        energy = np.linspace(1.0, 120.0, 1191)
        widths = np.gradient(energy)
        spectrum = np.zeros_like(energy)
        source_bin = np.argmin(np.abs(energy - 60.0))
        spectrum[source_bin] = 1.0 / widths[source_bin]
        free_scatter = single_scatter_spectrum(
            energy, spectrum, 0.3, 0.2, 0.1, 30.0, lambda e: np.full_like(e, 0.3))
        bound_scatter = single_scatter_spectrum(
            energy, spectrum, 0.3, 0.2, 0.1, 30.0, lambda e: np.full_like(e, 0.3),
            atomic_number=74)
        self.assertLess(np.sum(bound_scatter * widths), 0.5 * np.sum(free_scatter * widths))

        cos_min = np.cos(np.deg2rad(15.0))
        dc = (1.0 - cos_min) / 200000
        cosine = cos_min + (np.arange(200000) + 0.5) * dc
        escape = (np.exp(-0.03) - np.exp(-0.03 / cosine)) / (0.3 / cosine - 0.3)
        expected = np.sum(0.2 * escape * incoherent_angular_density(60.0, cosine, atomic_number=74) * dc)
        self.assertAlmostEqual(np.sum(bound_scatter * widths) / expected, 1.0, places=5)

    def test_coherent_rayleigh_form_factor_and_elastic_cone_transport(self):
        for z in (13, 29, 42, 74, 82):
            self.assertAlmostEqual(float(coherent_form_factor(z, 0.0)), float(z), places=5)
            self.assertTrue(np.all(np.diff(
                coherent_form_factor(z, np.geomspace(1e-4, 50.0, 50))) <= 0))

        # Unit normalization of coherent_angular_density over [-1, 1] using s = sin(theta/2)
        nodes, weights = np.polynomial.legendre.leggauss(128)
        s = 0.5 * (nodes + 1.0)
        cosines = 1.0 - 2.0 * s**2
        jacobian_weights = 2.0 * s * weights
        energies = np.array([10.0, 30.0, 60.0, 100.0, 225.0])
        for z in (None, 13, 29, 74):
            density = coherent_angular_density(energies[:, None], cosines, atomic_number=z)
            np.testing.assert_allclose(density @ jacobian_weights, np.ones(energies.size), rtol=2e-5)

        # Elastic scattering preserves photon energy bin-by-bin, and F(q, Z)^2 forward peaking
        # concentrates significantly more scatter into a 30-deg cone than unpolarized Thomson.
        energy = np.array([20.0, 60.0, 120.0])
        spectrum = np.array([0.0, 10.0, 0.0])
        thomson_coh = coherent_scatter_spectrum(
            energy, spectrum, 0.5, 0.1, 1e-7, 30.0, atomic_number=None)
        bound_coh = coherent_scatter_spectrum(
            energy, spectrum, 0.5, 0.1, 1e-7, 30.0, atomic_number=74)
        self.assertEqual(thomson_coh[0], 0.0)
        self.assertEqual(thomson_coh[2], 0.0)
        self.assertEqual(bound_coh[0], 0.0)
        self.assertEqual(bound_coh[2], 0.0)
        self.assertGreater(bound_coh[1], 5.0 * thomson_coh[1])


if __name__ == "__main__":
    unittest.main()
