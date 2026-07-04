import apply_filter_and_fluorescence as aff
import spekpy as sp
import numpy as np
import pandas as pd

def validate_fluorescence():
    # Load physical data
    master_df = aff.load_data("pyne_edpl", "attenuation_coefficients_long.csv")
    from pathlib import Path
    edpl_path = Path("pyne_edpl")
    
    # Elements to validate: Cu (29), Mo (42), Ag (47), Ta (73)
    elements = ["Cu", "Mo", "Ag", "Ta"]
    thickness_cm = 0.005 # 50 um
    
    # 100 kVp spectrum to excite all of them (Ta K-edge is ~67 keV)
    s = sp.Spek(kvp=100, th=12)
    energies, fluxes = s.get_spectrum(edges=True)
    
    spectrum_df = pd.DataFrame({
        'Energy_KeV': energies,
        'Photons_per_s': fluxes
    })
    
    results = []
    
    for el in elements:
        print(f"\n--- Validating {el} ---")
        z, sym = aff.get_element_properties(el)
        density = aff.DENSITIES.get(sym, 1.0)
        
        # We need to manually trace the physics steps in process_single_layer
        # to find the K-shell photoabsorption rate, and the emitted fluorescence rate
        
        thick_mm = thickness_cm * 10
        geom_params = {'radius_mm': 25.0, 'distance_mm': 1000.0, 'beam_angle_deg': 24.0}
        
        # Call process_single_layer
        Phi_out, Phi_fluoro, _, trans_factor = aff.process_single_layer(
            spectrum_df['Energy_KeV'].values,
            spectrum_df['Photons_per_s'].values,
            el,
            thick_mm,
            master_df,
            edpl_path,
            geom_params,
            calc_fluorescence=True,
            calc_scatter=False
        )
        
        # The sum of fluorescence photons
        total_fluor = Phi_fluoro.sum() if Phi_fluoro is not None else 0
        
        # Theoretical values from the same data the engine uses
        from fluorescence_k_data import FLUOR_K
        if z in FLUOR_K:
            f_data = FLUOR_K[z]
            wk = f_data['yield']
            ka = f_data['Ka']
            kb = f_data['Kb']
            kb_r = f_data['Kb_ratio']
        else:
            wk = float(z**4) / (33**4 + z**4)
            ka = 0.0102 * (z - 1)**2
            kb = ka * 1.12
            kb_r = 0.12
            
        print(f"Computed Fluorescence Yield (wk): {wk:.3f}")
        print(f"Computed K-alpha Energy: {ka:.2f} keV")
        print(f"Computed K-beta Energy: {kb:.2f} keV")
        print(f"Total escaping fluorescence photons/s: {total_fluor:.4e}")
        
        # Check primary absorption
        absorbed = spectrum_df['Photons_per_s'].sum() - Phi_out.sum()
        print(f"Total absorbed primary photons/s: {absorbed:.4e}")
        print(f"Apparent escape ratio (Fluor / Absorbed): {total_fluor / absorbed:.4e}")
        
        results.append({
            'Element': sym,
            'Z': z,
            'wK': wk,
            'E_Ka': ka,
            'E_Kb': kb,
            'Ratio_Ka_Kb': kb_r,
            'Total_Fluor': total_fluor,
            'Esc_Ratio': total_fluor / absorbed
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("validation_results.csv", index=False)
    
if __name__ == "__main__":
    validate_fluorescence()
