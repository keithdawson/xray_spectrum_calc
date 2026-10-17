import unittest
import numpy as np
import pandas as pd
from source_spectrum import generate_spekpy_spectrum, spectrum_normalization


class TestSourceNormalization(unittest.TestCase):
    def test_current_and_inverse_square_scaling(self):
        settings = dict(kvp=100, th=12, targ='W', dk=.5, tube_window='Aluminum (1mm)')
        energy, near = generate_spekpy_spectrum(**settings, tube_current_ma=1, distance_mm=1000)
        _, far = generate_spekpy_spectrum(**settings, tube_current_ma=1, distance_mm=2000)
        _, doubled = generate_spekpy_spectrum(**settings, tube_current_ma=2, distance_mm=1000)
        self.assertTrue(np.all(np.diff(energy) > 0))
        np.testing.assert_allclose(far, near / 4, rtol=1e-12)
        np.testing.assert_allclose(doubled, near * 2, rtol=1e-12)

    def test_csv_normalization_preserved(self):
        self.assertEqual(spectrum_normalization(pd.DataFrame({'Relative_Intensity': [1]})), 'relative/keV')
        self.assertEqual(spectrum_normalization(pd.DataFrame({'Photons_per_s': [1]})), 'photons/s/keV')
        frame = pd.DataFrame({'Photons_per_s': [1], 'Normalization': ['photons/cm2/s/keV']})
        self.assertEqual(spectrum_normalization(frame), 'photons/cm2/s/keV')


if __name__ == '__main__':
    unittest.main()
