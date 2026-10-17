"""Analytic checks for the primary-beam high-energy fluence optimizer."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import emission_optimizer as optimizer
from apply_filter_and_fluorescence import DENSITIES


class EmissionOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.att_path = self.folder / 'attenuation_coefficients_long.csv'
        pd.DataFrame({
            'Element': [29, 29], 'Energy_KeV': [1.0, 100.0],
            'Mu_over_rho': [2.0, 2.0],
        }).to_csv(self.att_path, index=False)

    def run_with_spectrum(self, energies, fluxes, **settings):
        source = Mock()
        source.get_spectrum.return_value = (np.array(energies), np.array(fluxes))
        with patch.object(optimizer, 'DATA_DIR', self.folder), \
                patch.object(optimizer.sp, 'Spek', return_value=source), \
                patch('apply_filter_and_fluorescence.PRACTICAL_FILTER_MATERIALS', ['Cu']):
            result = optimizer.run_emission_optimizer(kvp=100, dk=40, **settings)
        source.get_spectrum.assert_called_once_with(edges=False)
        return result

    def test_threshold_splits_a_bin_without_losing_energy(self):
        low, high = optimizer._band_energy_weights([40.0, 80.0], 40.0, 57.0)
        np.testing.assert_allclose(low, [1424.5, 0.0])
        np.testing.assert_allclose(high, [175.5, 3200.0])
        np.testing.assert_allclose(low + high, np.array([40, 80]) * 40)

    def test_gray_filter_matches_beer_lambert_analytic_score(self):
        result = self.run_with_spectrum([40, 80], [3, 7], max_thickness_mm=0.1)
        # An energy-independent attenuator has the same transmission in
        # both bands regardless of the spectrum or the threshold crossing.
        row = result.iloc[0]
        self.assertAlmostEqual(row.Thickness_mm, 0.1)
        transmission = np.exp(-2.0 * DENSITIES['Cu'] * 0.1 / 10.0)
        self.assertAlmostEqual(row.Kept_High_Pct, transmission * 100)
        self.assertAlmostEqual(row.Removed_Low_Pct, (1 - transmission) * 100)
        self.assertAlmostEqual(row.Score, (1 - transmission) * transmission ** 2 * 1e6)

    def test_noninteger_thickness_limit_is_evaluated_and_reported_exactly(self):
        result = self.run_with_spectrum([40, 80], [3, 7], max_thickness_mm=0.015)
        self.assertAlmostEqual(result.iloc[0].Thickness_mm, 0.015)

    def test_empty_energy_band_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'both bands'):
            self.run_with_spectrum([20, 40], [1, 0], max_thickness_mm=0.1)

    def test_invalid_inputs_are_rejected_before_generating_a_spectrum(self):
        for settings in ({'kvp': 57}, {'dk': 0}, {'max_thickness_mm': 0.001},
                         {'th': float('nan')}, {'threshold_kev': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                optimizer.run_emission_optimizer(**settings)

    def test_cache_tracks_anode_settings_and_data(self):
        settings = dict(kvp=100, max_thickness_mm=0.1, th=12, targ='W', dk=0.1,
                        threshold_kev=57)
        base = optimizer._cache_path(settings, self.att_path, ['Cu'], DENSITIES)
        for key, value in dict(kvp=110, max_thickness_mm=0.2, th=15, targ='Mo',
                               dk=0.2, threshold_kev=60).items():
            with self.subTest(setting=key):
                changed = optimizer._cache_path(dict(settings, **{key: value}),
                                                self.att_path, ['Cu'], DENSITIES)
                self.assertNotEqual(base, changed)
        self.att_path.write_text(self.att_path.read_text() + '\n')
        self.assertNotEqual(base, optimizer._cache_path(settings, self.att_path,
                                                       ['Cu'], DENSITIES))

    def test_custom_filter_materials_subset(self):
        source = Mock()
        source.get_spectrum.return_value = (np.array([40.0, 80.0]), np.array([3.0, 7.0]))
        with patch.object(optimizer, 'DATA_DIR', self.folder), \
                patch.object(optimizer.sp, 'Spek', return_value=source):
            result = optimizer.run_emission_optimizer(
                kvp=100, dk=40, max_thickness_mm=0.1, filter_materials=['Cu', 'Cu']
            )
        self.assertEqual(list(result['Filter_Material']), ['Cu'])
        for bad_mats in ([], ['NotAnElement']):
            with self.subTest(bad_mats=bad_mats), self.assertRaises(ValueError):
                optimizer.run_emission_optimizer(kvp=100, dk=40, filter_materials=bad_mats)


if __name__ == '__main__':
    unittest.main()
