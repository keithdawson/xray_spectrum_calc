import unittest
import numpy as np
import pandas as pd
import os
import shutil
import sys

# Ensure we can import the module from the current directory
sys.path.append(os.getcwd())
try:
    from apply_filter_and_flourescence import process_single_layer
except ImportError:
    # Fallback if running from a different directory structure
    pass

class TestFluorescencePhysics(unittest.TestCase):
    def setUp(self):
        self.att_file = 'test_att_physics.csv'
        self.edpl_folder = 'test_edpl_physics'
        os.makedirs(self.edpl_folder, exist_ok=True)
        
        # Setup Energy Grid (keV)
        # 1 to 100 keV in 1 keV steps
        self.E_kev = np.linspace(1, 100, 100) 
        self.E_mev = self.E_kev / 1000.0
        
        # Setup Attenuation Data (Mo, Z=42)
        # Mo K-edge is approx 20 keV.
        # We create a simplified model:
        # Below 20 keV: mu = 10 cm2/g
        # Above 20 keV: mu = 50 cm2/g (representing the edge jump)
        mu_vals = np.where(self.E_kev < 20.0, 10.0, 50.0)
        
        self.att_df = pd.DataFrame({
            'Energy_MeV': self.E_mev,
            'Element': [42]*100,
            'Mu_over_rho': mu_vals,
            'Mu_en_over_rho': mu_vals
        })
        self.att_df['Energy_KeV'] = self.E_kev
        
        # Setup EPDL Data
        # tauK (photoelectric K-shell cross section)
        # Physically, this is 0 below the K-edge.
        # Above 20 keV, let's say tauK = 40 cm2/g 
        # (meaning 80% of interactions above the edge are K-shell photoelectric)
        tauK_vals = np.where(self.E_kev < 20.0, 0.0, 40.0)
        
        df_edpl = pd.DataFrame({
            'E_keV': self.E_kev,
            'tauK_over_rho_cm2_per_g': tauK_vals
        })
        
        # The script looks for files matching *ZA042*.csv for Z=42
        df_edpl.to_csv(os.path.join(self.edpl_folder, 'test_ZA042_extracted.csv'), index=False)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.edpl_folder):
            shutil.rmtree(self.edpl_folder)
        # The att_df is passed as an object, so no file cleanup needed for it unless we wrote it to disk in main logic

    def test_no_fluorescence_below_edge(self):
        """Verify no fluorescence is produced when incident energy is below K-edge."""
        # Incident Spectrum: Monochromatic delta function at 15 keV (below 20 keV edge)
        # Units: Photons/s/keV (Spectral Density)
        phi_in = np.zeros_like(self.E_kev)
        idx_15 = np.abs(self.E_kev - 15.0).argmin()
        phi_in[idx_15] = 1.0e6 
        
        # Geometry: Large acceptance to ensure we'd see it if it were there
        geom = {'radius_mm': 100, 'distance_mm': 100} 
        
        # Run calculation
        # Thickness 0.01 mm
        phi_out = process_single_layer(
            self.E_kev, phi_in, 'Mo', 0.01, self.att_df, self.edpl_folder, geom
        )
        
        # Check K-alpha (17.48 keV) and K-beta (19.61 keV) bins
        idx_Ka = np.abs(self.E_kev - 17.48).argmin()
        idx_Kb = np.abs(self.E_kev - 19.61).argmin()
        
        # Since incident energy (15) < K-edge (20), tauK is 0.
        # Fluorescence should be exactly 0.0.
        self.assertAlmostEqual(phi_out[idx_Ka], 0.0, msg="Found fluorescence below K-edge (Ka bin)")
        self.assertAlmostEqual(phi_out[idx_Kb], 0.0, msg="Found fluorescence below K-edge (Kb bin)")
        
        # Verify transmission at 15 keV
        # T = exp(-mu * rho * t) = exp(-10 * 10.28 * 0.001) = exp(-0.1028) ~ 0.9023
        # t_cm = 0.01 mm / 10 = 0.001 cm
        expected_trans = 1.0e6 * np.exp(-10.0 * 10.28 * 0.001)
        self.assertAlmostEqual(phi_out[idx_15], expected_trans, delta=expected_trans*0.001)

    def test_fluorescence_above_edge(self):
        """Verify fluorescence is produced when incident energy is above K-edge."""
        # Incident Spectrum: Monochromatic delta function at 30 keV (above 20 keV edge)
        phi_in = np.zeros_like(self.E_kev)
        idx_30 = np.abs(self.E_kev - 30.0).argmin()
        phi_in[idx_30] = 1.0e6
        
        # Geometry: "Half sphere" acceptance (2pi steradians)
        # theta = atan(10000/1) ~ 90 deg. frac ~ 0.5
        geom = {'radius_mm': 10000, 'distance_mm': 1} 
        
        # Run calculation
        # Thickness 0.001 mm (very thin to minimize self-absorption complexity)
        phi_out = process_single_layer(
            self.E_kev, phi_in, 'Mo', 0.001, self.att_df, self.edpl_folder, geom
        )
        
        idx_Ka = np.abs(self.E_kev - 17.48).argmin()
        idx_Kb = np.abs(self.E_kev - 19.61).argmin()
        
        # Check peaks exist
        self.assertGreater(phi_out[idx_Ka], 0.0, "Missing K-alpha peak above K-edge")
        self.assertGreater(phi_out[idx_Kb], 0.0, "Missing K-beta peak above K-edge")
        
        # --- Physics Check ---
        # 1. Absorption Probability: P_abs = 1 - exp(-mu * rho * t)
        #    mu=50, rho=10.28, t=0.0001 cm
        #    exponent = 50 * 10.28 * 0.0001 = 0.0514
        #    P_abs = 1 - exp(-0.0514) approx 0.0501
        p_abs = 1.0 - np.exp(-50.0 * 10.28 * 0.0001)
        
        # 2. K-shell Fraction: tauK / mu_tot = 40 / 50 = 0.8
        frac_k = 0.8
        
        # 3. Vacancies Created = Flux * P_abs * Frac_K
        #    1e6 * 0.0501 * 0.8 approx 40,080 vacancies/s
        vacancies = 1.0e6 * p_abs * frac_k
        
        # 4. Yield (Mo) = 0.765
        #    Emission = 40,080 * 0.765 approx 30,661 photons/s
        emission = vacancies * 0.765
        
        # 5. Geometric Acceptance approx 0.5
        #    Detected = 30,661 * 0.5 approx 15,330 photons/s
        expected_detected = emission * 0.5
        
        # 6. Result in bins
        #    The script adds (Emission * Branching * Geom * Escape) / dE to the bin.
        #    dE = 1 keV. Escape prob ~ 1.0 for thin filter.
        #    Sum of Ka and Kb bins should equal total detected.
        
        measured_detected = (phi_out[idx_Ka] + phi_out[idx_Kb]) * 1.0 # * dE
        
        # Allow 5% tolerance for numerical integration/approximations
        self.assertAlmostEqual(measured_detected, expected_detected, delta=expected_detected*0.05, 
                               msg=f"Fluorescence magnitude mismatch. Expected ~{expected_detected:.0f}, Got {measured_detected:.0f}")

if __name__ == '__main__':
    unittest.main()