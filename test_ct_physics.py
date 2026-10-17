"""Analytic detector/noise and optimizer regressions; no SpekPy required."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import ct_optimizer
import ct_physics
from ct_materials import MATERIAL_DEFINITIONS


def ideal_response(energies, material, thickness):
    energies = np.asarray(energies, dtype=float)
    return np.ones_like(energies), energies, energies**2


class DetectorPhysicsTests(unittest.TestCase):
    def test_compound_poisson_moments_and_bin_widths(self):
        # Each bin is 20 keV wide: 40 and 60 incident photons, half interacting.
        # Each interaction deposits 5 or 15 keV. Independent Poisson moments
        # give mean=550 keV, variance=7250 keV^2, not variance=mean.
        energies = np.array([10.0, 30.0])
        coefficients = (np.full(2, np.log(2)), np.full(2, np.log(2) / 2), 10.0)
        with patch.object(ct_physics, 'material_coefficients', return_value=coefficients):
            result = ct_physics.calculate_contrast(
                pd.DataFrame({'Energy_KeV': energies, 'Photons_per_s': [2.0, 3.0]}),
                'A', 0.0, 'B', 1.0, 'Detector', 1.0,
            )
        self.assertAlmostEqual(result['signal_A'], 550)
        self.assertAlmostEqual(result['variance_A'], 7250)
        self.assertAlmostEqual(result['signal_B'], 275)
        self.assertAlmostEqual(result['variance_B'], 3625)
        self.assertAlmostEqual(result['cnr_per_sqrt_exposure'], 275 / np.sqrt(10875))
        self.assertAlmostEqual(result['contrast'], 0.5)

    def test_zero_screen_and_invalid_thickness(self):
        energies = np.array([20.0, 40.0])
        for response in ct_physics.detector_response(energies, 'CsI', 0.0):
            np.testing.assert_array_equal(response, [0.0, 0.0])
        for thickness in (-1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                ct_physics.attenuation_exponent('Cu', energies, thickness)
        with self.assertRaises(ValueError):
            ct_physics._muen_over_mu('missing detector', energies)

    def test_compound_mass_fractions_and_shared_filter_attenuation(self):
        from apply_filter_and_fluorescence import process_single_layer
        for _, composition in MATERIAL_DEFINITIONS.values():
            self.assertAlmostEqual(sum(composition.values()), 1.0)
        energies = np.array([20.0, 30.0, 60.0, 100.0])
        photons = np.ones(4)
        for material in ('Cu', 'Mo', 'Air'):
            thickness = 0.1
            _, _, _, transmission = process_single_layer(
                energies, photons, material, thickness, ct_physics._attenuation_table(),
                'pyne_edpl', {'beam_angle_deg': 30.0},
                calc_fluorescence=False, calc_scatter=False,
            )
            np.testing.assert_allclose(
                transmission, np.exp(-ct_physics.attenuation_exponent(material, energies, thickness)),
                rtol=1e-12,
            )

    def test_nist_absorption_edge_is_not_smoothed_by_spekpy(self):
        # NIST iodine K edge is 33.1694 keV: the tabulated post-edge jump
        # should remain large on a fine grid rather than smeared over keV.
        below, above = ct_physics.material_coefficients('I', [33.1693, 33.1698])[0]
        self.assertGreater(above / below, 4.0)


class CnrOptimizerTests(unittest.TestCase):
    @staticmethod
    def optimize(photons=None, limit=5.0):
        if photons is None:
            photons = np.array([10.0, 1.0])
        return ct_optimizer.run_filter_optimization(
            np.array([10.0, 20.0]), photons, 'A', 1.0, 'B', 1.0,
            'Shadow', 0.0, 'Shadow', 0.0, 'Detector', 1.0,
            time_penalty_limit=limit, filter_materials=['Filter'],
        )

    @staticmethod
    def exponent(material, energies, thickness):
        coefficients = {'A': [0.0, 0.0], 'B': [0.0, np.log(2)],
                        'Shadow': [0.0, 0.0], 'Filter': [1.0, 0.0]}
        return np.asarray(coefficients[material]) * thickness

    def test_fixed_capacity_cnr_uses_two_second_moments(self):
        with patch.object(ct_optimizer, 'detector_response', side_effect=ideal_response), \
                patch.object(ct_optimizer, 'attenuation_exponent', side_effect=self.exponent):
            sweep, optimum = self.optimize(limit=1.0)
        row = optimum.iloc[0]
        # Width=10: air signal=1200; A signal=1200, B=1100;
        # variances=14000 and 12000. This differs from sqrt(signal) noise.
        self.assertEqual(row['Thickness_mm'], 0.0)
        self.assertEqual(row['Time_Penalty'], 1.0)
        self.assertAlmostEqual(row['Dynamic_CNR'], 100 / np.sqrt(1200 * 26000))
        self.assertIn('Variance_A', sweep.columns)

    def test_capacity_cnr_scale_invariance_and_boundary_optimum(self):
        with patch.object(ct_optimizer, 'detector_response', side_effect=ideal_response), \
                patch.object(ct_optimizer, 'attenuation_exponent', side_effect=self.exponent):
            _, optimum = self.optimize()
            _, brighter = self.optimize(np.array([70.0, 7.0]))
        row = optimum.iloc[0]
        self.assertAlmostEqual(row['Time_Penalty'], 5.0)
        # Air signal at limit=240; hard-bin signal=200, so the soft bin
        # falls from 1000 to 40: thickness=ln(25) mm.
        self.assertAlmostEqual(row['Thickness_mm'], np.log(25), places=8)
        self.assertAlmostEqual(row['Dynamic_CNR'], brighter.iloc[0]['Dynamic_CNR'])

    def test_no_filter_endpoint_can_be_optimal(self):
        def exponent(material, energies, thickness):
            coefficient = {'A': [0.0, 0.0], 'B': [np.log(2), np.log(2)],
                           'Shadow': [0.0, 0.0], 'Filter': [1.0, 0.1]}
            return np.asarray(coefficient[material]) * thickness
        with patch.object(ct_optimizer, 'detector_response', side_effect=ideal_response), \
                patch.object(ct_optimizer, 'attenuation_exponent', side_effect=exponent):
            _, optimum = self.optimize()
        self.assertEqual(optimum.iloc[0]['Thickness_mm'], 0.0)

    def test_invalid_inputs(self):
        for limit in (0.5, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.optimize(limit=limit)
        with self.assertRaises(ValueError):
            self.optimize(photons=np.array([-1.0, 2.0]))

    def test_contrast_and_optimizer_share_the_same_detector_model(self):
        energies = np.array([20.0, 30.0, 50.0, 80.0])
        photons = np.array([40.0, 50.0, 30.0, 10.0])
        spectrum = pd.DataFrame({'Energy_KeV': energies, 'Photons_per_s': photons})
        contrast = ct_physics.calculate_contrast(spectrum, 'FR4', 1.0, 'Cu', 0.035, 'Gadox', 0.2)
        air = ct_physics.calculate_contrast(spectrum, 'Air', 0.0, 'Air', 0.0, 'Gadox', 0.2)
        _, optimum = ct_optimizer.run_filter_optimization(
            energies, photons, 'FR4', 1.0, 'Cu', 0.035, 'Air', 0.0, 'Air', 0.0,
            'Gadox', 0.2, time_penalty_limit=1.0, filter_materials=['Cu'],
        )
        row = optimum.iloc[0]
        for path in ('A', 'B'):
            self.assertAlmostEqual(row[f'Signal_{path}'], contrast[f'signal_{path}'])
            self.assertAlmostEqual(row[f'Variance_{path}'], contrast[f'variance_{path}'])
        self.assertAlmostEqual(row['Signal_Max'], air['signal_A'])
        self.assertAlmostEqual(
            row['Dynamic_CNR'], contrast['cnr_per_sqrt_exposure'] / np.sqrt(air['signal_A'])
        )

    def test_include_secondary_fog_penalizes_high_fog_candidate(self):
        energies = np.array([10.0, 20.0])
        photons = np.array([10.0, 1.0])

        def exponent(material, e_grid, thickness):
            coefficients = {
                'A': [0.0, 0.0],
                'B': [0.0, np.log(2)],
                'Shadow': [0.0, 0.0],
                'Clean': [1.0, 0.0],
                'Foggy': [1.0, 0.0],
            }
            return np.asarray(coefficients[material]) * thickness

        def mock_layer(e_grid, phi_in, material, thickness, table, folder, geometry,
                       calc_fluorescence, calc_scatter):
            trans = np.exp(-exponent(material, e_grid, thickness))
            if material == 'Foggy' and thickness > 0:
                fog = np.array([0.0, 0.2 * (1.0 - np.exp(-thickness))])
            else:
                fog = np.zeros_like(e_grid)
            return phi_in * trans + fog, fog, np.zeros_like(e_grid), trans

        with patch.object(ct_optimizer, 'detector_response', side_effect=ideal_response), \
                patch.object(ct_optimizer, 'attenuation_exponent', side_effect=exponent), \
                patch.object(ct_optimizer.aff, 'process_single_layer', side_effect=mock_layer):
            _, opt_no_fog = ct_optimizer.run_filter_optimization(
                energies, photons, 'A', 1.0, 'B', 1.0, 'Shadow', 0.0, 'Shadow', 0.0,
                'Detector', 1.0, time_penalty_limit=5.0, filter_materials=['Clean', 'Foggy'],
                include_secondary_fog=False,
            )
            _, opt_with_fog = ct_optimizer.run_filter_optimization(
                energies, photons, 'A', 1.0, 'B', 1.0, 'Shadow', 0.0, 'Shadow', 0.0,
                'Detector', 1.0, time_penalty_limit=5.0, filter_materials=['Clean', 'Foggy'],
                include_secondary_fog=True, geom_params={'beam_angle_deg': 45.0},
            )
        # Without secondary fog, Clean and Foggy have identical primary attenuation and CNR.
        np.testing.assert_allclose(
            opt_no_fog.loc[opt_no_fog['Filter_Material'] == 'Clean', 'Dynamic_CNR'].values,
            opt_no_fog.loc[opt_no_fog['Filter_Material'] == 'Foggy', 'Dynamic_CNR'].values,
        )
        # With secondary fog enabled, Foggy has higher Signal_Max/Variance and lower Dynamic_CNR.
        clean_row = opt_with_fog[opt_with_fog['Filter_Material'] == 'Clean'].iloc[0]
        foggy_row = opt_with_fog[opt_with_fog['Filter_Material'] == 'Foggy'].iloc[0]
        self.assertEqual(opt_with_fog.iloc[0]['Filter_Material'], 'Clean')
        self.assertGreater(clean_row['Dynamic_CNR'], foggy_row['Dynamic_CNR'])
        self.assertGreater(foggy_row['Secondary_Fog_Pct'], 0.0)
        self.assertEqual(clean_row['Secondary_Fog_Pct'], 0.0)


class FluorescenceOptimizerTests(unittest.TestCase):
    def test_downstream_attenuation_is_applied_after_fluorescence_generation(self):
        energies = np.array([10.0, 20.0, 40.0])
        primary = np.array([0.0, 0.0, 10.0])
        fluorescence = np.array([0.0, 2.0, 0.0])

        def layer(energies, photons, material, thickness, table, folder, geometry,
                  calc_fluorescence, calc_scatter):
            if material == 'First':
                np.testing.assert_array_equal(photons, primary)
                return primary + fluorescence, fluorescence, np.zeros(3), np.ones(3)
            if thickness == 0:
                return photons, np.zeros(3), np.zeros(3), np.ones(3)
            if photons[2] == 0:
                # Air absorption belongs after this excitation, not before it.
                np.testing.assert_array_equal(photons, fluorescence)
            emitted = np.array([0.1 if photons[2] > 0 else 0.2, 0.0, 0.0])
            return photons * 0.8 + emitted, emitted, np.zeros(3), np.full(3, 0.8)

        with patch.object(ct_optimizer.aff, 'process_single_layer', side_effect=layer):
            sweep, _ = ct_optimizer.run_fluorescence_optimization(
                energies, primary, 'First', 0.1, candidate_materials=['Second'],
                downstream_transmission=[0.0, 0.1, 1.0],
            )
        selected = sweep[sweep['Thickness_mm'] > 0]
        # Air transmits 10% of 20-keV fluorescence and none of the new
        # 10-keV daughters: reference 60, candidate 48 keV -> 20% eliminated.
        np.testing.assert_allclose(selected['F_Total'], 48.0)
        np.testing.assert_allclose(selected['F_elim'], 0.2)
        np.testing.assert_allclose(selected['P_Total'], 6400.0)
        np.testing.assert_allclose(selected['P_kept'], 0.8)
        for invalid in ([1, 1], [0, -0.1, 1], [0, 1.1, 1], [0, np.nan, 1]):
            with self.assertRaises(ValueError):
                ct_optimizer.run_fluorescence_optimization(
                    energies, primary, 'First', 0.1, downstream_transmission=invalid,
                )

    def test_upstream_fluorescence_excites_next_layer_and_geometry_is_passed(self):
        energies = np.array([10.0, 20.0, 40.0])
        primary = np.array([0.0, 0.0, 10.0])
        first_fluorescence = np.array([0.0, 2.0, 0.0])
        seen_geometry = []

        def layer(energies, photons, material, thickness, table, folder, geometry,
                  calc_fluorescence, calc_scatter):
            seen_geometry.append(geometry['beam_angle_deg'])
            if material == 'First':
                return primary + first_fluorescence, first_fluorescence, np.zeros(3), np.ones(3)
            if thickness == 0.0:
                return photons, np.zeros(3), np.zeros(3), np.ones(3)
            if photons[2] > 0:
                emitted = np.array([0.1, 0.0, 0.0])
            else:
                emitted = np.array([0.2, 0.0, 0.0]) if calc_fluorescence else np.zeros(3)
            return photons * 0.8 + emitted, emitted, np.zeros(3), np.full(3, 0.8)

        with patch.object(ct_optimizer.aff, 'process_single_layer', side_effect=layer):
            sweep, _ = ct_optimizer.run_fluorescence_optimization(
                energies, primary, 'First', 0.1, candidate_materials=['Second'],
                geom_params={'beam_angle_deg': 72.0},
            )
        # Retained first fluorescence: 1.6*20*15=480; main/cascade
        # fluorescence: (0.1+0.2)*10*10=30; total=510 keV.
        np.testing.assert_allclose(sweep.loc[sweep['Thickness_mm'] > 0, 'F_Total'], 510.0)
        np.testing.assert_allclose(sweep.loc[sweep['Thickness_mm'] > 0, 'P_kept'], 0.8)
        np.testing.assert_allclose(sweep.loc[sweep['Thickness_mm'] > 0, 'F_elim'], 0.15)
        self.assertTrue(all(angle == 72.0 for angle in seen_geometry))

    def test_filter_that_adds_fluorescence_loses_to_no_filter(self):
        energies = np.array([10.0, 20.0, 40.0])
        primary = np.array([0.0, 0.0, 10.0])
        fluorescence = np.array([0.0, 2.0, 0.0])

        def layer(energies, photons, material, thickness, table, folder, geometry,
                  calc_fluorescence, calc_scatter):
            if material == 'First':
                return primary + fluorescence, fluorescence, np.zeros(3), np.ones(3)
            transmission = np.exp(-thickness)
            # A hypothetical downstream foil adds more line energy than it removes.
            emitted = np.array([100 * (1 - transmission), 0.0, 0.0])
            return photons * transmission + emitted, emitted, np.zeros(3), np.full(3, transmission)

        with patch.object(ct_optimizer.aff, 'process_single_layer', side_effect=layer):
            sweep, optimum = ct_optimizer.run_fluorescence_optimization(
                energies, primary, 'First', 0.1, candidate_materials=['Second'],
            )
        self.assertTrue((sweep.loc[sweep['Thickness_mm'] > 0, 'F_elim'] < 0).all())
        self.assertEqual(optimum.iloc[0]['Thickness_mm'], 0.0)
        self.assertEqual(optimum.iloc[0]['F_elim'], 0.0)

    def test_include_scatter_penalizes_scatter_producing_secondary_filter(self):
        energies = np.array([10.0, 20.0, 40.0])
        primary = np.array([0.0, 0.0, 10.0])
        fluorescence = np.array([0.0, 2.0, 0.0])
        scatter1 = np.array([0.0, 0.0, 0.2])

        def layer(energies, photons, material, thickness, table, folder, geometry,
                  calc_fluorescence, calc_scatter):
            if material == 'First':
                s_out = scatter1 if calc_scatter else np.zeros(3)
                return primary + fluorescence + s_out, fluorescence, s_out, np.ones(3)
            if thickness == 0:
                return photons, np.zeros(3), np.zeros(3), np.ones(3)
            # CleanFilter absorbs 80% of 20-keV fluorescence with minimal scatter;
            # ScatterFilter absorbs 85% of fluorescence but generates heavy scatter from primary.
            if material == 'CleanFilter':
                trans = np.array([0.8, 0.2, 0.9])
                s_out = np.array([0.0, 0.0, 0.02]) if (calc_scatter and photons[2] > 0) else np.zeros(3)
            else:
                trans = np.array([0.8, 0.15, 0.9])
                s_out = np.array([0.0, 0.0, 1.5]) if (calc_scatter and photons[2] > 0) else np.zeros(3)
            return photons * trans + s_out, np.zeros(3), s_out, trans

        with patch.object(ct_optimizer.aff, 'process_single_layer', side_effect=layer):
            _, top_fluor_only = ct_optimizer.run_fluorescence_optimization(
                energies, primary, 'First', 0.1,
                candidate_materials=['CleanFilter', 'ScatterFilter'],
                include_scatter=False,
            )
            _, top_with_scatter = ct_optimizer.run_fluorescence_optimization(
                energies, primary, 'First', 0.1,
                candidate_materials=['CleanFilter', 'ScatterFilter'],
                include_scatter=True,
            )
        self.assertEqual(top_fluor_only.iloc[0]['Filter_Material'], 'ScatterFilter')
        self.assertEqual(top_with_scatter.iloc[0]['Filter_Material'], 'CleanFilter')
        self.assertGreater(top_with_scatter.iloc[0]['Secondary_elim'], 0.5)

    def test_multi_layer_upstream_filters_cascade(self):
        energies = np.array([10.0, 20.0, 40.0])
        primary = np.array([0.0, 0.0, 10.0])

        def layer(energies, photons, material, thickness, table, folder, geometry,
                  calc_fluorescence, calc_scatter):
            if thickness == 0:
                return photons, np.zeros(3), np.zeros(3), np.ones(3)
            if material == 'W_Layer':
                # Emits 20-keV fluorescence from 40-keV primary
                fl = np.array([0.0, 2.0, 0.0]) if photons[2] > 0 else np.zeros(3)
                trans = np.array([1.0, 1.0, 0.5])
                return photons * trans + fl, fl, np.zeros(3), trans
            if material == 'Sn_Layer':
                # Absorbs 20-keV fluorescence and emits 10-keV daughter fluorescence
                fl = np.array([1.0, 0.0, 0.0]) if photons[1] > 0 else np.zeros(3)
                trans = np.array([1.0, 0.1, 0.9])
                return photons * trans + fl, fl, np.zeros(3), trans
            if material == 'Cu_Layer':
                # Absorbs 10-keV daughter fluorescence while keeping 40-keV primary
                trans = np.array([0.1, 0.8, 0.95])
                return photons * trans, np.zeros(3), np.zeros(3), trans
            raise AssertionError(f'Unexpected material: {material}')

        with patch.object(ct_optimizer.aff, 'process_single_layer', side_effect=layer):
            sweep, top = ct_optimizer.run_fluorescence_optimization(
                energies, primary,
                candidate_materials=['Cu_Layer'],
                upstream_filters=[
                    {'material': 'W_Layer', 'thickness': 0.2},
                    {'material': 'Sn_Layer', 'thickness': 0.1},
                ],
            )
        active = sweep[sweep['Thickness_mm'] > 0].iloc[0]
        # Upstream exit has 0.2 at 20 keV (power=0.2*20*15=60) and 1.0 at 10 keV (power=1.0*10*10=100) -> F0=160.
        # Cu_Layer transmits 10% of 10 keV (10) and 80% of 20 keV (48) -> F=58, F_elim = 1 - 58/160 = 0.6375.
        self.assertAlmostEqual(active['F_Total'], 58.0)
        self.assertAlmostEqual(active['F_elim'], 1.0 - 58.0 / 160.0)
        self.assertAlmostEqual(active['P_kept'], 0.95)
        self.assertGreater(top.iloc[0]['Thickness_mm'], 0.0)
        with self.assertRaises(ValueError):
            ct_optimizer.run_fluorescence_optimization(energies, primary, upstream_filters=[])
        with self.assertRaises(ValueError):
            ct_optimizer.run_fluorescence_optimization(energies, primary)


if __name__ == '__main__':
    unittest.main(verbosity=2)
