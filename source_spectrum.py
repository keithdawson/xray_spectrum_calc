"""SpekPy source normalization shared by the calculator and optimizers."""
import numpy as np
import spekpy


def generate_spekpy_spectrum(kvp, th, targ, dk, tube_current_ma, distance_mm, tube_window):
    """Return bin centres and photons/cm2/s/keV at the specified distance.

    SpekPy gives photons/cm2/keV for an exposure in mAs and distance in cm.
    A one-second exposure at I mA uses I mAs. Air transport is added by
    the caller separately; source distance already includes inverse square.
    """
    if (not np.isfinite(distance_mm) or distance_mm <= 0
            or not np.isfinite(tube_current_ma) or tube_current_ma <= 0):
        raise ValueError('Source distance and tube current must be finite and positive.')
    source = spekpy.Spek(kvp=kvp, th=th, targ=targ, dk=dk,
                         z=distance_mm / 10.0, mas=tube_current_ma)
    if tube_window == 'Beryllium (2mm)':
        source.filter('Be', 2.0)
    elif tube_window == 'Aluminum (1mm)':
        source.filter('Al', 1.0)
    elif tube_window != 'None':
        raise ValueError(f'Unknown tube window: {tube_window}')
    return source.get_spectrum()


def spectrum_normalization(frame, generated=False):
    if generated:
        return 'photons/cm2/s/keV'
    if 'Normalization' in frame.columns and len(frame):
        value = str(frame['Normalization'].iloc[0])
        if value in ('photons/cm2/s/keV', 'photons/s/keV', 'relative/keV'):
            return value
    if 'Relative_Intensity' in frame.columns:
        return 'relative/keV'
    return 'photons/s/keV'
