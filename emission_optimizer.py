import numpy as np
import pandas as pd
import spekpy as sp
import os

def _best_per_material(df):
    """Collapses a full thickness sweep down to the single best-scoring
    thickness for each material, ranked by score."""
    best_idx = df.groupby('Filter_Material')['Score'].idxmax()
    return df.loc[best_idx].sort_values(by='Score', ascending=False).reset_index(drop=True)

def run_emission_optimizer(kvp=220, max_thickness_mm=2.0, th=12, targ='W', dk=0.1):
    """
    Sweeps practical filter materials (elements available as thin foils/discs)
    and thicknesses (0.01mm resolution) to find the best filter that removes
    < 57 keV energy flux while preserving >= 57 keV energy flux.

    Returns one row per material: its best-scoring thickness.
    """
    from apply_filter_and_fluorescence import PRACTICAL_FILTER_MATERIALS

    cache_file = f"emission_opt_results_{kvp}kV_{max_thickness_mm}mm_dk{dk}.csv"
    if os.path.exists(cache_file):
        df_cached = pd.read_csv(cache_file)
        # Older caches may contain the full 92-element sweep; keep only
        # materials that are physically usable as filters.
        df_cached = df_cached[df_cached['Filter_Material'].isin(PRACTICAL_FILTER_MATERIALS)]
        return _best_per_material(df_cached)
        
    print(f"Running Emission Optimizer for {kvp} kVp...")
    
    # 1. Generate Baseline Spectrum
    s = sp.Spek(kvp=kvp, th=th, targ=targ, dk=dk)
    energies, fluxes = s.get_spectrum(edges=True)
    
    # Energy flux (Power) = E * Phi(E)
    power_spectrum = energies * fluxes
    
    # Create boolean masks for region integration
    # "Power kept at 57 and above" -> energies >= 57.0
    mask_high = energies >= 57.0
    mask_low = energies < 57.0
    
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    
    power_base_high = trapz(power_spectrum[mask_high], energies[mask_high])
    power_base_low = trapz(power_spectrum[mask_low], energies[mask_low])
    
    # 2. Load attenuation data
    from apply_filter_and_fluorescence import SYM_TO_Z, DENSITIES
    att_df = pd.read_csv("attenuation_coefficients_long.csv")
    
    # 3. Setup Thickness Sweep
    # from 0.01 mm to max_thickness_mm in 0.01 mm steps
    num_steps = int(round(max_thickness_mm / 0.01))
    thicknesses_mm = np.linspace(0.01, max_thickness_mm, num_steps)
    thicknesses_cm = thicknesses_mm / 10.0
    
    results = []
    
    # 4. Sweep Elements (practical foil materials only)
    for sym in PRACTICAL_FILTER_MATERIALS:
        z = SYM_TO_Z[sym]
        density = DENSITIES.get(sym, 1.0)
        elem_att = att_df[att_df['Element'] == z]
        if elem_att.empty:
            continue
            
        mu_tot = np.interp(energies, elem_att['Energy_KeV'], elem_att['Mu_over_rho'])
        
        # Calculate array of rho * t for all thicknesses
        # Shape: (num_thicknesses, 1)
        rho_t_array = (density * thicknesses_cm).reshape(-1, 1)
        
        # Transmission factor shape: (num_thicknesses, num_energies)
        # Broadcasting: (N, 1) * (1, E) -> (N, E)
        trans_factors = np.exp(-rho_t_array * mu_tot.reshape(1, -1))
        
        # Multiply baseline power spectrum by transmission
        filtered_power = trans_factors * power_spectrum.reshape(1, -1)
        
        # Integrate Low (<57) and High (>=57) for all thicknesses
        # Using trapz along the energy axis (axis=1)
        power_filt_high = trapz(filtered_power[:, mask_high], energies[mask_high], axis=1)
        power_filt_low = trapz(filtered_power[:, mask_low], energies[mask_low], axis=1)
        
        # Calculate Metrics
        pct_kept_high = (power_filt_high / power_base_high) * 100.0
        pct_kept_low = (power_filt_low / power_base_low) * 100.0
        pct_removed_low = 100.0 - pct_kept_low
        
        # Score = (Removed Low) * (Kept High)^2
        score = pct_removed_low * (pct_kept_high ** 2)
        
        for i, t_mm in enumerate(thicknesses_mm):
            results.append({
                'Filter_Material': sym,
                'Thickness_mm': round(t_mm, 2),
                'Removed_Low_Pct': pct_removed_low[i],
                'Kept_High_Pct': pct_kept_high[i],
                'Score': score[i]
            })
            
    # Compile Results
    df_res = pd.DataFrame(results)
    
    # Sort by Score descending
    df_res = df_res.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # Cache the full sweep so display logic can change without re-running
    df_res.to_csv(cache_file, index=False)

    return _best_per_material(df_res)
