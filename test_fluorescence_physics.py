import unittest
import numpy as np
import pandas as pd
import os
import shutil
import math
from apply_filter_and_fluorescence import (
    process_single_layer, run_stack, DENSITIES, calculate_fluorescence_escape,
    fluorescence_cone_escape, acceptance_cone_from_geometry, resolve_acceptance_angle_deg,
)
from fluorescence_k_data import FLUOR_K
from spectrum_utils import energy_bin_widths, integrate_spectrum, interpolate_attenuation, add_line_density

class TestFluorescencePhysics(unittest.TestCase):
    def setUp(self):
        self.att_file = 'test_att_physics.csv'
        self.edpl_folder = 'test_edpl_physics'
        os.makedirs(self.edpl_folder, exist_ok=True)
        
        # Energy Grid: 1 to 100 keV
        self.E_kev = np.linspace(1, 100, 100)
        self.E_mev = self.E_kev / 1000.0
        
        # We will mock Molybdenum (Z=42, K-edge ~20 keV, density ~10.28)
        # and Tungsten (Z=74, K-edge ~69 keV)
        
        # Mo Attenuation
        mu_mo = np.where(self.E_kev < 20.0, 10.0, 50.0)
        tauK_mo = np.where(self.E_kev < 20.0, 0.0, 40.0)
        
        # Cu Attenuation (Z=29, K-edge ~9 keV)
        mu_cu = np.where(self.E_kev < 9.0, 15.0, 60.0)
        tauK_cu = np.where(self.E_kev < 9.0, 0.0, 45.0)
        
        # W Attenuation (Z=74, K-edge ~69 keV)
        mu_w = np.where(self.E_kev < 69.0, 8.0, 40.0)
        tauK_w = np.where(self.E_kev < 69.0, 0.0, 35.0)

        # Build attenuation DF
        df_list = []
        df_list.append(pd.DataFrame({'Energy_MeV': self.E_mev, 'Energy_KeV': self.E_kev, 'Element': 42, 'Mu_over_rho': mu_mo, 'Mu_en_over_rho': mu_mo}))
        df_list.append(pd.DataFrame({'Energy_MeV': self.E_mev, 'Energy_KeV': self.E_kev, 'Element': 29, 'Mu_over_rho': mu_cu, 'Mu_en_over_rho': mu_cu}))
        df_list.append(pd.DataFrame({'Energy_MeV': self.E_mev, 'Energy_KeV': self.E_kev, 'Element': 74, 'Mu_over_rho': mu_w, 'Mu_en_over_rho': mu_w}))
        self.att_df = pd.concat(df_list)
        self.att_df.to_csv(self.att_file, index=False)
        
        # Build EPDL files
        pd.DataFrame({'E_keV': self.E_kev, 'tauK_over_rho_cm2_per_g': tauK_mo}).to_csv(os.path.join(self.edpl_folder, 'test_ZA042.csv'), index=False)
        pd.DataFrame({'E_keV': self.E_kev, 'tauK_over_rho_cm2_per_g': tauK_cu}).to_csv(os.path.join(self.edpl_folder, 'test_ZA029.csv'), index=False)
        pd.DataFrame({'E_keV': self.E_kev, 'tauK_over_rho_cm2_per_g': tauK_w}).to_csv(os.path.join(self.edpl_folder, 'test_ZA074.csv'), index=False)

    def tearDown(self):
        if os.path.exists(self.edpl_folder):
            shutil.rmtree(self.edpl_folder)
        if os.path.exists(self.att_file):
            os.remove(self.att_file)

    def test_1_k_edge_threshold(self):
        '''Test 1: Energy Dependence Test (Below and Above K-Edge)'''
        geom = {'beam_angle_deg': 180.0} # 50% acceptance
        
        # Below edge (15 keV)
        phi_in_below = np.zeros_like(self.E_kev)
        idx_15 = np.abs(self.E_kev - 15.0).argmin()
        phi_in_below[idx_15] = 1e6
        phi_out_below, fl_below, _, _ = process_single_layer(self.E_kev, phi_in_below, 'Mo', 1.0, self.att_df, self.edpl_folder, geom)
        self.assertEqual(np.sum(fl_below), 0.0, "Fluorescence should be zero below K-edge.")
        
        # Above edge (21 keV)
        phi_in_above = np.zeros_like(self.E_kev)
        idx_21 = np.abs(self.E_kev - 21.0).argmin()
        phi_in_above[idx_21] = 1e6
        phi_out_above, fl_above, _, _ = process_single_layer(self.E_kev, phi_in_above, 'Mo', 1.0, self.att_df, self.edpl_folder, geom)
        self.assertGreater(np.sum(fl_above), 0.0, "Fluorescence should be generated above K-edge.")

    def test_2_yield_limit_conservation(self):
        '''Test 2: Conservation of Energy (Yield limit)'''
        geom = {'beam_angle_deg': 180.0} # 50% solid angle -> frac = 0.5
        
        # Mo parameters from apply_filter_and_fluorescence.py:
        # yield: 0.765, Ka: 17.48, Kb: 19.61
        phi_in = np.zeros_like(self.E_kev)
        idx_25 = np.abs(self.E_kev - 25.0).argmin()
        phi_in[idx_25] = 1e6 # 1 million photons at 25 keV
        
        thickness_mm = 0.01
        t_cm = thickness_mm / 10.0
        rho = DENSITIES['Mo'] # 10.22
        
        phi_out, fl_out, _, _ = process_single_layer(self.E_kev, phi_in, 'Mo', thickness_mm, self.att_df, self.edpl_folder, geom)
        
        # Calculate expected limits analytically
        # Mu_tot at 25 keV is 50.0, tauK is 40.0
        mu_in = 50.0 * rho
        tau_in = 40.0 * rho
        
        total_interactions = 1e6 * (1 - np.exp(-mu_in * t_cm))
        photo_k = total_interactions * (40.0 / 50.0)
        generated_fl = photo_k * 0.765
        
        escaping_fl = np.sum(fl_out)
        max_possible_exiting = generated_fl * 0.5
        
        self.assertGreater(escaping_fl, 0.0)
        self.assertLess(escaping_fl, max_possible_exiting, "Escaping fluorescence exceeded generated bounds! Energy conservation violated.")
        
    def test_3_self_attenuation_thickness_sweep(self):
        '''Test 3: Thickness Dependence Test'''
        geom = {'beam_angle_deg': 180.0}
        phi_in = np.zeros_like(self.E_kev)
        phi_in[np.abs(self.E_kev - 25.0).argmin()] = 1e6
        
        thicknesses = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        fl_results = []
        for t in thicknesses:
            _, fl_out, _, _ = process_single_layer(self.E_kev, phi_in, 'Mo', t, self.att_df, self.edpl_folder, geom)
            fl_results.append(np.sum(fl_out))
            
        peak_idx = np.argmax(fl_results)
        self.assertGreater(peak_idx, 0, "Fluorescence should increase with thickness initially.")
        self.assertLess(peak_idx, len(thicknesses)-1, "Fluorescence should decay at massive thicknesses due to self-attenuation.")
        
    def test_4_geometric_solid_angle(self):
        '''Test 4: Geometric Solid Angle Validation'''
        phi_in = np.zeros_like(self.E_kev)
        phi_in[np.abs(self.E_kev - 25.0).argmin()] = 1e6
        
        _, fl_180, _, _ = process_single_layer(self.E_kev, phi_in, 'Mo', 0.1, self.att_df, self.edpl_folder, {'beam_angle_deg': 180.0})
        _, fl_90, _, _ = process_single_layer(self.E_kev, phi_in, 'Mo', 0.1, self.att_df, self.edpl_folder, {'beam_angle_deg': 90.0})
        _, fl_0, _, _ = process_single_layer(self.E_kev, phi_in, 'Mo', 0.1, self.att_df, self.edpl_folder, {'beam_angle_deg': 0.0})
        
        self.assertEqual(np.sum(fl_0), 0.0)
        self.assertGreater(fl_180.sum(), fl_90.sum())
        # Oblique rays travel farther through the foil. Solid angle alone
        # overestimates hemisphere escape for finite optical thickness.
        self.assertLess(fl_180.sum() / fl_90.sum(), 0.5 / 0.1464466)
        thin_180 = fluorescence_cone_escape(np.array([50.0]), 10.0, 1e-10, 180.0)
        thin_90 = fluorescence_cone_escape(np.array([50.0]), 10.0, 1e-10, 90.0)
        self.assertAlmostEqual(float(thin_180[0] / thin_90[0]), 0.5 / 0.1464466094, places=5)

    def test_depth_and_angle_integrals_against_quadrature(self):
        from scipy.integrate import quad
        for mu_in, mu_out, depth in [(10.0, 10.0 + 1e-12, 0.2), (50., 10., 0.1), (1000., 1., 1.)]:
            expected = quad(lambda x: np.exp(-mu_in*x - mu_out*(depth-x)), 0, depth,
                            epsabs=1e-14)[0]
            self.assertAlmostEqual(float(calculate_fluorescence_escape(mu_in, mu_out, depth)), expected, places=12)
        expected = quad(lambda c: quad(
            lambda x: np.exp(-50*x - 10/c*(0.1-x)), 0, 0.1)[0] / 2,
            np.cos(np.deg2rad(30)), 1)[0]
        actual = fluorescence_cone_escape(np.array([50.]), 10., 0.1, 60.)[0]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_last_bin_and_line_energy_are_conserved(self):
        phi = np.zeros_like(self.E_kev)
        phi[-1] = 1e6
        thickness = 1e-7
        _, fluorescence, _, _ = process_single_layer(
            self.E_kev, phi, 'Mo', thickness, self.att_df, self.edpl_folder,
            {'beam_angle_deg': 30}, calc_scatter=False)
        atom = FLUOR_K[42]
        expected = 1e6 * 40 * DENSITIES['Mo'] * thickness / 10 * atom['yield'] * (1-np.cos(np.deg2rad(15))) / 2
        self.assertAlmostEqual(fluorescence.sum() / expected, 1, places=5)
        expected_mean = (atom['Ka'] + atom['Kb_ratio'] * atom['Kb']) / (1 + atom['Kb_ratio'])
        # Residual differential self-absorption is O(thickness).
        self.assertAlmostEqual(float(np.dot(fluorescence, self.E_kev) / fluorescence.sum()), expected_mean, delta=1e-6)
        energy = np.array([1., 8., 17., 19., 30.])
        line = np.zeros_like(energy)
        add_line_density(line, energy, atom['Ka'], 100.)
        self.assertAlmostEqual(integrate_spectrum(energy, line), 100., places=12)
        self.assertAlmostEqual(integrate_spectrum(energy, line * energy), 100. * atom['Ka'], places=10)

    def test_lines_outside_grid_are_not_moved_to_boundary(self):
        energy = np.array([30., 40., 50.])
        _, fluorescence, _, _ = process_single_layer(
            energy, np.ones(3), 'Mo', 0.1, self.att_df, self.edpl_folder,
            {'beam_angle_deg': 30}, calc_scatter=False)
        np.testing.assert_array_equal(fluorescence, 0)

    def test_invalid_spectrum_and_negative_thickness_rejected(self):
        for energy, phi, thickness in [([1.,1.], [1.,1.], .1), ([1.,2.], [-1.,1.], .1),
                                       ([1.,2.], [1.,1.], -.1)]:
            with self.assertRaises(ValueError):
                process_single_layer(energy, phi, 'Mo', thickness, self.att_df,
                                     self.edpl_folder, {'beam_angle_deg': 30})

    def test_log_interpolation_and_bin_integrals(self):
        np.testing.assert_allclose(interpolate_attenuation([2.,4.], [1.,8.], [64.,1.]), [16.,4.])
        with self.assertRaises(ValueError):
            interpolate_attenuation([9.], [1.,8.], [64.,1.])
        energy = np.array([1.,2.,4.])
        self.assertEqual(integrate_spectrum(energy, np.ones(3)), 4.5)

    def test_zero_thickness_and_partial_rate_bounds(self):
        phi = np.full_like(self.E_kev, 1e5)
        output, fl, scatter, trans = process_single_layer(
            self.E_kev, phi, 'Mo', 0., self.att_df, self.edpl_folder, {'beam_angle_deg': 30})
        np.testing.assert_array_equal(output, phi)
        np.testing.assert_array_equal(fl + scatter, 0)
        np.testing.assert_array_equal(trans, 1)
        # Inconsistent cross-section sources must not create photons.
        low_mu = self.att_df.copy()
        low_mu['Mu_over_rho'] = 1.0
        output, _, _, _ = process_single_layer(
            self.E_kev, phi, 'Mo', .01, low_mu, self.edpl_folder, {'beam_angle_deg': 180})
        self.assertLessEqual(integrate_spectrum(self.E_kev, output), integrate_spectrum(self.E_kev, phi))
        
    def test_5_cascading_multi_layer(self):
        '''Test 5: Cascading Multi-Layer Fluorescence'''
        phi_in_df = pd.DataFrame({'Energy_keV': self.E_kev, 'Relative_Intensity': 0.0})
        phi_in_df.loc[phi_in_df['Energy_keV'] == 75.0, 'Relative_Intensity'] = 1e8
        spec_file = 'test_spec.csv'
        phi_in_df.to_csv(spec_file, index=False)
        
        out_file = 'test_out.csv'
        
        filters = [
            {'material': 'W', 'thickness': 0.1},
            {'material': 'Cu', 'thickness': 0.05}
        ]
        
        geom = {'beam_angle_deg': 180.0}
        
        run_stack(spec_file, filters, self.att_file, self.edpl_folder, geom, out_file, save_fluorescence=True)
        
        res_df = pd.read_csv(out_file)
        fluoro_df = pd.read_csv('test_out_fluoro.csv')
        
        cu_ka_idx = np.abs(fluoro_df['Energy_KeV'] - 8.05).argmin()
        cu_fl_intensity = fluoro_df['Photons_per_s'].iloc[cu_ka_idx]
        
        self.assertGreater(cu_fl_intensity, 0.0, "Cu fluorescence was not triggered by W fluorescence cascade!")
        
        if os.path.exists(spec_file): os.remove(spec_file)
        if os.path.exists(out_file): os.remove(out_file)
        if os.path.exists('test_out_fluoro.csv'): os.remove('test_out_fluoro.csv')
        if os.path.exists('test_out_scatter.csv'): os.remove('test_out_scatter.csv')

    def test_acceptance_cone_from_physical_geometry(self):
        # D = 2*L -> half-angle = 45 deg -> full angle = 90 deg
        self.assertAlmostEqual(acceptance_cone_from_geometry(100.0, 200.0), 90.0, places=9)
        self.assertAlmostEqual(acceptance_cone_from_geometry(100.0, 0.0), 0.0, places=9)
        self.assertAlmostEqual(
            resolve_acceptance_angle_deg({'filter_to_detector_mm': 100.0, 'detector_diameter_mm': 200.0}),
            90.0, places=9
        )
        # Per-layer distance override changes the subtended cone angle
        self.assertAlmostEqual(
            resolve_acceptance_angle_deg(
                {'filter_to_detector_mm': 500.0, 'detector_diameter_mm': 200.0},
                {'filter_to_detector_mm': 100.0}
            ),
            90.0, places=9
        )
        for bad_dist, bad_diam in [(0.0, 100.0), (-5.0, 100.0), (100.0, -1.0), (np.nan, 100.0)]:
            with self.assertRaises(ValueError):
                acceptance_cone_from_geometry(bad_dist, bad_diam)

if __name__ == '__main__':
    unittest.main()
