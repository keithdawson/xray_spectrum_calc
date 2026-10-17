"""Nominal CT materials: densities in g/cm^3, compositions by mass.

FR4/solder are bulk approximations. Actual boards, fillers and screen packing
fractions need measured properties for quantitative predictions.
"""


def _mass_fractions(stoichiometric_masses):
    total = sum(stoichiometric_masses.values())
    return {z: mass / total for z, mass in stoichiometric_masses.items()}


MATERIAL_DEFINITIONS = {
    'FR4': (1.85, {14: 0.280, 8: 0.385, 6: 0.291, 1: 0.025, 7: 0.019}),
    'Steel': (7.90, {26: 0.69, 24: 0.19, 28: 0.10, 25: 0.02}),
    'Gadox': (7.34, _mass_fractions({64: 2 * 157.25, 8: 2 * 15.999, 16: 32.06})),
    'CsI': (4.51, _mass_fractions({55: 132.90545196, 53: 126.90447})),
    'Leaded Solder': (8.40, {50: 0.63, 82: 0.37}),
    'Leadfree Solder': (7.38, {50: 0.965, 47: 0.03, 29: 0.005}),
}

CT_MATERIALS = ['FR4', 'Cu', 'Steel', 'Al', 'Ge', 'Sn', 'Pb', 'Leaded Solder', 'Leadfree Solder', 'Air']
SCINTILLATORS = ['Gadox', 'CsI']
