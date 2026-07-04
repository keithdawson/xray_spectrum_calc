import numpy as np
import pandas as pd
import spekpy
from scipy import optimize

def run_filter_optimization(energies, raw_photons, target_a_mat, target_a_thick, target_b_mat, target_b_thick, shadow1_mat, shadow1_thick, shadow2_mat, shadow2_thick, scint_mat, scint_thick, path_multiplier=1.0, time_penalty_limit=5.0, extra_filters=None, filter_materials=None):
    """
    Evaluates different filter materials and thicknesses to maximize Dynamic-Range-Limited CNR 
    while tracking Time Penalty based on detector saturation (S_max through air).
    Uses continuous optimization to find the exact optimal thickness up to the time penalty limit.
    """
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    mu_data = spekpy.SpekTools.MuData('pene_mu.dat')
    
    # Pre-calculate base attenuations (Target A, Target B, Shadow, Scintillator)
    mu_t_A = mu_data.get_mu_t(target_a_mat, energies, target_a_thick) * path_multiplier
    mu_t_B = mu_data.get_mu_t(target_b_mat, energies, target_b_thick) * path_multiplier
    mu_t_shadow1 = mu_data.get_mu_t(shadow1_mat, energies, shadow1_thick)
    mu_t_shadow2 = mu_data.get_mu_t(shadow2_mat, energies, shadow2_thick)
    mu_t_shadow = (mu_t_shadow1 + mu_t_shadow2) * path_multiplier
    
    mu_t_scint = mu_data.get_mu_t(scint_mat, energies, scint_thick)
    scint_prob = 1.0 - np.exp(-mu_t_scint)
    
    # Path Attenuations
    path1_atten = np.exp(-(mu_t_A + mu_t_shadow))
    path2_atten = np.exp(-(mu_t_B + mu_t_shadow))
    
    # Unfiltered S_max (Air path: The beam that saturates the detector)
    unfiltered_photons_max = raw_photons * scint_prob
    unfiltered_s_max = trapz(unfiltered_photons_max * energies, energies)
    
    if unfiltered_s_max <= 0:
        raise ValueError("Unfiltered signal is zero. Increase tube kVp.")

    if filter_materials is None:
        filter_materials = ['Al', 'Fe', 'Cu', 'Mo', 'Ag', 'Sn', 'Ta', 'W', 'Pb']
    else:
        filter_materials = list(filter_materials)

    if extra_filters:
        for f in extra_filters:
            if f not in filter_materials:
                filter_materials.append(f)
    
    def evaluate_thickness(t, f_mat):
        if t == 0:
            filter_atten = np.ones_like(energies)
        else:
            mu_t_filter = mu_data.get_mu_t(f_mat, energies, t)
            filter_atten = np.exp(-mu_t_filter)
        
        sig_max_photons = raw_photons * filter_atten * scint_prob
        sig1_photons = raw_photons * filter_atten * path1_atten * scint_prob
        sig2_photons = raw_photons * filter_atten * path2_atten * scint_prob
        
        s_max = trapz(sig_max_photons * energies, energies)
        s_A = trapz(sig1_photons * energies, energies)
        s_B = trapz(sig2_photons * energies, energies)
        
        if s_A > 1e-150 and s_max > 1e-150:
            delta_s = abs(s_A - s_B)
            cnr_dr = delta_s / np.sqrt(s_A * s_max)
            time_penalty = unfiltered_s_max / s_max
        else:
            cnr_dr = 0.0
            time_penalty = float('inf')
            
        return s_max, s_A, s_B, cnr_dr, time_penalty
        
    def find_thickness_for_tp(f_mat, target_tp):
        def tp_diff(t):
            _, _, _, _, tp = evaluate_thickness(t, f_mat)
            return tp - target_tp
        if tp_diff(0.0) >= 0:
            return 0.0
        if tp_diff(100.0) < 0:
            return 100.0
        try:
            res = optimize.root_scalar(tp_diff, bracket=[0.0, 100.0], method='brentq')
            return res.root
        except Exception:
            return 100.0

    results = []
    top_results = []
    
    for f_mat in filter_materials:
        t_3 = find_thickness_for_tp(f_mat, 3.0)
        t_15 = find_thickness_for_tp(f_mat, 15.0)
        
        if t_3 >= t_15 or t_3 == 0.0:
            mat_thicknesses = np.linspace(0.00, 3.0, 91)
        else:
            t_low = np.linspace(0.00, t_3, 61)
            t_high = np.linspace(t_3, t_15, 31)[1:]
            mat_thicknesses = np.concatenate([t_low, t_high])
            
        # First, generate the discrete sweep for plotting
        for t in mat_thicknesses:
            s_max, s_A, s_B, cnr_dr, time_penalty = evaluate_thickness(t, f_mat)
            results.append({
                'Filter_Material': f_mat,
                'Thickness_mm': t,
                'Signal_Max': s_max,
                'Signal_A': s_A,
                'Signal_B': s_B,
                'Dynamic_CNR': cnr_dr,
                'Time_Penalty': time_penalty
            })
            
        # Second, perform continuous optimization for the top result
        # 1. Find the thickness boundary `t_limit` where time_penalty == time_penalty_limit
        def root_func(t):
            _, _, _, _, tp = evaluate_thickness(t, f_mat)
            return tp - time_penalty_limit
            
        try:
            # We search up to 100mm. If it doesn't cross the limit by 100mm, we cap at 100mm.
            res_root = optimize.root_scalar(root_func, bracket=[0.0, 100.0], method='brentq')
            t_limit = res_root.root
        except ValueError:
            # If the limit isn't reached even at 100mm, or if time penalty exceeds limit even at 0 (unlikely)
            t_limit = 100.0

        # 2. Find the exact thickness that maximizes CNR between [0, t_limit]
        def obj_func(t):
            _, _, _, cnr, _ = evaluate_thickness(t, f_mat)
            return -cnr # minimize negative CNR
            
        res_opt = optimize.minimize_scalar(obj_func, bounds=(0.0, t_limit), method='bounded')
        t_opt = res_opt.x
        
        # Calculate final stats at the perfectly optimized thickness
        s_max_opt, s_A_opt, s_B_opt, cnr_opt, tp_opt = evaluate_thickness(t_opt, f_mat)
        top_results.append({
            'Filter_Material': f_mat,
            'Thickness_mm': t_opt,
            'Signal_Max': s_max_opt,
            'Signal_A': s_A_opt,
            'Signal_B': s_B_opt,
            'Dynamic_CNR': cnr_opt,
            'Time_Penalty': tp_opt
        })
            
    df = pd.DataFrame(results)
    top_per_mat = pd.DataFrame(top_results).sort_values(by='Dynamic_CNR', ascending=False)
    
    return df, top_per_mat

import apply_filter_and_fluorescence as aff

def run_fluorescence_optimization(energies, raw_photons, mat1, thick1, candidate_materials=None):
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    att_df = pd.read_csv('attenuation_coefficients_long.csv')
    att_df['Energy_KeV'] = att_df['Energy_MeV'] * 1000.0
    edpl_folder = 'pyne_edpl'
    
    geom_params = {
        'beam_angle_deg': 40.0,
        'radius_mm': 5.0,
        'distance_mm': 100.0
    }
    
    # 1. Base Layer (Filter 1)
    phi_tot1, phi_fluor1, _, trans_factor1 = aff.process_single_layer(
        energies, raw_photons, mat1, thick1, att_df, edpl_folder, geom_params, calc_fluorescence=True, calc_scatter=False
    )
    phi_trans1 = phi_tot1 - phi_fluor1
    
    # Base Power and Fluorescence Power
    P0 = trapz(phi_trans1 * energies, energies)
    F0 = trapz(phi_fluor1 * energies, energies)
    
    if P0 <= 0 or F0 <= 0:
        return pd.DataFrame(), pd.DataFrame()
        
    if candidate_materials is None:
        candidate_materials = ['Al', 'Cu', 'Mo', 'Ag', 'Sn', 'W', 'Pb']
    thicknesses = np.linspace(0.01, 2.0, 50)
    
    results = []
    
    for mat2 in candidate_materials:
        for t2 in thicknesses:
            phi_tot2, phi_fluor_from_main2, _, _ = aff.process_single_layer(
                energies, phi_trans1, mat2, t2, att_df, edpl_folder, geom_params, calc_fluorescence=True, calc_scatter=False
            )
            phi_trans2 = phi_tot2 - phi_fluor_from_main2
            
            phi_fluor1_attenuated_total, _, _, _ = aff.process_single_layer(
                energies, phi_fluor1, mat2, t2, att_df, edpl_folder, geom_params, calc_fluorescence=False, calc_scatter=False
            )
            
            P = trapz(phi_trans2 * energies, energies)
            total_F_spectrum = phi_fluor1_attenuated_total + phi_fluor_from_main2
            F = trapz(total_F_spectrum * energies, energies)
            
            p_kept = P / P0 if P0 > 0 else 0
            f_elim = 1.0 - (F / F0) if F0 > 0 else 0
            f_elim = max(f_elim, 0.0)
            
            score = f_elim * (p_kept ** 2)
            
            results.append({
                'Filter_Material': mat2,
                'Thickness_mm': t2,
                'P_kept': p_kept,
                'F_elim': f_elim,
                'Score': score,
                'P_Total': P,
                'F_Total': F
            })
            
    df = pd.DataFrame(results)
    valid_df = df[df['P_kept'] > 0.01]
    if valid_df.empty:
        valid_df = df
        
    top_per_mat = valid_df.loc[valid_df.groupby('Filter_Material')['Score'].idxmax()].sort_values(by='Score', ascending=False)
    
    return df, top_per_mat
