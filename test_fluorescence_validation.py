"""Integration checks using shipped cross sections and SpekPy bin centres.

These are conservation/reference checks, not experimental validation of the
plane-slab model. Photon densities are integrated over bin widths throughout.
"""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import spekpy

import apply_filter_and_fluorescence as aff
from fluorescence_k_data import FLUOR_K
from spectrum_utils import integrate_spectrum

ROOT = Path(__file__).resolve().parent


class TestPhysicalDataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attenuation = pd.read_csv(ROOT / 'attenuation_coefficients_long.csv')
        source = spekpy.Spek(kvp=100, th=12, dk=0.1)
        source.filter('Al', 1.)
        cls.energies, cls.incident = source.get_spectrum()

    def test_real_material_conservation_and_shell_yield(self):
        for material in ['Cu', 'Mo', 'Ag', 'Ta']:
            with self.subTest(material=material):
                energy, incident = self.energies, self.incident
                total, fluorescence, scatter, transmission = aff.process_single_layer(
                    energy, incident, material, .05, self.attenuation,
                    ROOT / 'pyne_edpl', {'beam_angle_deg': 24.})
                removed_primary = integrate_spectrum(energy, incident * (1-transmission))
                n_fluor = integrate_spectrum(energy, fluorescence)
                n_scatter = integrate_spectrum(energy, scatter)
                yield_k = FLUOR_K[aff.SYM_TO_Z[material]]['yield']
                acceptance = (1-np.cos(np.deg2rad(12))) / 2
                self.assertGreater(n_fluor, 0)
                self.assertGreater(n_scatter, 0)
                self.assertLessEqual(n_fluor, removed_primary * yield_k * acceptance)
                self.assertLessEqual(integrate_spectrum(energy, total), integrate_spectrum(energy, incident))
                self.assertLessEqual(integrate_spectrum(energy, total * energy), integrate_spectrum(energy, incident * energy))
                np.testing.assert_allclose(total, incident*transmission + fluorescence + scatter)
                print(f'{material}: fluorescence={n_fluor:.6g}, scatter={n_scatter:.6g}, removed primary={removed_primary:.6g}')

    def test_cross_section_support_and_edge_anchors(self):
        for z in range(11, 93):
            data = pd.read_csv(ROOT / 'pyne_edpl' / f'ZA{z:03d}_extracted.csv')
            self.assertTrue(np.all(np.diff(data.E_keV) > 0), f'Z={z}: duplicate edge anchors')
            self.assertEqual(data.E_keV.max(), 300.)
            self.assertGreater(data.tauK_over_rho_cm2_per_g.iloc[-1], 0.)
            self.assertTrue(np.all(data.loc[data.E_keV < FLUOR_K[z]['K_edge'], 'tauK_over_rho_cm2_per_g'] == 0))
        with self.assertRaisesRegex(ValueError, '300'):
            aff.process_single_layer(np.array([299., 301.]), np.ones(2), 'Mo', .1,
                                     self.attenuation, ROOT / 'pyne_edpl', {'beam_angle_deg': 30.})

    def test_low_z_alpha_only_emitters(self):
        for material in ['Na', 'Mg']:
            z = aff.SYM_TO_Z[material]
            self.assertEqual(FLUOR_K[z]['Kb_ratio'], 0.)
            energy = np.arange(1., 10.01, .01)
            incident = np.where(energy > 3., 1., 0.)
            _, fluorescence, _, _ = aff.process_single_layer(
                energy, incident, material, .01, self.attenuation,
                ROOT / 'pyne_edpl', {'beam_angle_deg': 30.}, calc_scatter=False)
            self.assertGreater(integrate_spectrum(energy, fluorescence), 0.)
            mean_energy = integrate_spectrum(energy, fluorescence * energy) / integrate_spectrum(energy, fluorescence)
            self.assertAlmostEqual(mean_energy, FLUOR_K[z]['Ka'], places=10)

    def test_nist_anchor_and_no_runtime_generator_dependencies(self):
        # Published NIST Cu value at 100 keV is 0.4584 cm2/g.
        copper = self.attenuation[self.attenuation.Element == 29]
        self.assertAlmostEqual(copper.loc[copper.Energy_KeV == 100., 'Mu_over_rho'].iloc[0], .4584)
        self.assertEqual(self.attenuation.Element.nunique(), 92)


if __name__ == '__main__':
    unittest.main()
