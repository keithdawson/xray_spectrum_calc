import spekpy
import os

def initialize_custom_materials():
    """
    Injects custom material definitions into SpekPy's internal composition dictionary
    so they can be referenced by name in attenuation calculations.
    """
    
    # FR4: ~60% SiO2, 40% Epoxy. Density ~ 1.85 g/cm^3
    spekpy.SpekTools.make_composition_def(
        'FR4',
        1.85,
        [(14, 0.280), (8, 0.385), (6, 0.291), (1, 0.025), (7, 0.019)],
        '',
        'PCB Substrate FR4 approximation'
    )
        
    # Steel (SS304): Fe (69%), Cr (19%), Ni (10%), Mn (2%). Density 7.9 g/cm^3
    spekpy.SpekTools.make_composition_def(
        'Steel',
        7.90,
        [(26, 0.69), (24, 0.19), (28, 0.10), (25, 0.02)],
        '',
        'Stainless Steel 304'
    )

    # Gadox Scintillator (Gd2O2S): Gd (83.1%), O (8.5%), S (8.5%). Density 7.34 g/cm^3
    spekpy.SpekTools.make_composition_def(
        'Gadox',
        7.34,
        [(64, 0.831), (8, 0.085), (16, 0.085)],
        '',
        'Gadolinium oxysulfide scintillator'
    )

    # CsI Scintillator: Cs (51.2%), I (48.8%). Density 4.51 g/cm^3
    spekpy.SpekTools.make_composition_def(
        'CsI',
        4.51,
        [(55, 0.512), (53, 0.488)],
        '',
        'Cesium Iodide scintillator'
    )

    # Leaded Solder (Sn63Pb37): Sn (63%), Pb (37%). Density ~8.40 g/cm^3
    spekpy.SpekTools.make_composition_def(
        'Leaded Solder',
        8.40,
        [(50, 0.63), (82, 0.37)],
        '',
        'Sn63Pb37 Leaded Solder'
    )

    # Leadfree Solder (SAC305): Sn (96.5%), Ag (3%), Cu (0.5%). Density ~7.38 g/cm^3
    spekpy.SpekTools.make_composition_def(
        'Leadfree Solder',
        7.38,
        [(50, 0.965), (47, 0.03), (29, 0.005)],
        '',
        'SAC305 Leadfree Solder'
    )

# Ensure materials are loaded on import
initialize_custom_materials()

CT_MATERIALS = ['FR4', 'Cu', 'Steel', 'Al', 'Ge', 'Sn', 'Pb', 'Leaded Solder', 'Leadfree Solder', 'Air']
SCINTILLATORS = ['Gadox', 'CsI']
