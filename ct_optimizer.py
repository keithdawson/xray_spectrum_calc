"""Primary-ray CNR and fluorescence-suppression filter searches."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from ct_physics import attenuation_exponent, detector_response, validate_spectrum
from spectrum_utils import energy_bin_widths, integrate_spectrum


def run_filter_optimization(energies, raw_photons, target_a_mat, target_a_thick, target_b_mat, target_b_thick, shadow1_mat, shadow1_thick, shadow2_mat, shadow2_thick, scint_mat, scint_thick, path_multiplier=1.0, time_penalty_limit=5.0, extra_filters=None, filter_materials=None):
    """Maximize idealized CNR at fixed air-path detector energy capacity.

    Let S be deposited-energy rate and V be its compound-Poisson variance
    rate. With air-path capacity Q, exposure is Q/S_air, and the CNR of
    independent A/B measurements is sqrt(Q)*abs(S_A-S_B)/sqrt(S_air*(V_A+V_B)).
    Dynamic_CNR returns this coefficient per sqrt(keV of capacity), not an
    absolute CNR. Time_Penalty is S_air(unfiltered)/S_air(filtered).

    The response matches calculate_contrast; it neglects event-energy/optical
    gain noise, electronic noise, object scatter, filter secondaries and CT
    reconstruction. A coarse scan plus local refinements includes endpoints
    and multiple resolved peaks; it is not a proof of a global optimum.
    """
    energies, raw_photons = validate_spectrum(energies, raw_photons)
    if not np.isfinite(path_multiplier) or path_multiplier <= 0:
        raise ValueError('Path multiplier must be finite and positive.')
    if not np.isfinite(time_penalty_limit) or time_penalty_limit < 1:
        raise ValueError('Time penalty limit must be finite and at least 1.')
    widths = energy_bin_widths(energies)
    _, mean_kernel, variance_kernel = detector_response(energies, scint_mat, scint_thick)
    shadow = path_multiplier * (
        attenuation_exponent(shadow1_mat, energies, shadow1_thick)
        + attenuation_exponent(shadow2_mat, energies, shadow2_thick)
    )
    path_a = np.exp(-shadow - path_multiplier * attenuation_exponent(target_a_mat, energies, target_a_thick))
    path_b = np.exp(-shadow - path_multiplier * attenuation_exponent(target_b_mat, energies, target_b_thick))
    mean_air = raw_photons * mean_kernel * widths
    variance_air = raw_photons * variance_kernel * widths
    unfiltered_s_max = np.sum(mean_air)
    if unfiltered_s_max <= 0:
        raise ValueError('Unfiltered detector signal is zero. Check spectrum and scintillator thickness.')

    materials = list(filter_materials) if filter_materials is not None else ['Al', 'Fe', 'Cu', 'Mo', 'Ag', 'Sn', 'Ta', 'W', 'Pb']
    for material in extra_filters or []:
        if material not in materials:
            materials.append(material)
    materials = list(dict.fromkeys(materials))
    columns = ['Filter_Material', 'Thickness_mm', 'Signal_Max', 'Signal_A', 'Signal_B',
               'Variance_A', 'Variance_B', 'Dynamic_CNR', 'Time_Penalty']
    results, top_results = [], []

    for material in materials:
        mu_per_mm = attenuation_exponent(material, energies, 1.0)

        def evaluate(thickness):
            transmission = np.exp(-mu_per_mm * thickness)
            filtered_mean = mean_air * transmission
            filtered_variance = variance_air * transmission
            s_air = float(np.sum(filtered_mean))
            s_a = float(np.dot(filtered_mean, path_a))
            s_b = float(np.dot(filtered_mean, path_b))
            v_a = float(np.dot(filtered_variance, path_a))
            v_b = float(np.dot(filtered_variance, path_b))
            time_penalty = unfiltered_s_max / s_air if s_air > 0 else np.inf
            denominator = np.sqrt(v_a + v_b) * np.sqrt(s_air)
            cnr = abs(s_a - s_b) / denominator if denominator > 0 else 0.0
            return {'Filter_Material': material, 'Thickness_mm': float(thickness),
                    'Signal_Max': s_air, 'Signal_A': s_a, 'Signal_B': s_b,
                    'Variance_A': v_a, 'Variance_B': v_b,
                    'Dynamic_CNR': cnr, 'Time_Penalty': time_penalty}

        def thickness_for_penalty(penalty):
            if penalty == 1.0:
                return 0.0
            # Find the boundary using signal instead of an overflowing ratio.
            threshold = unfiltered_s_max / penalty
            def difference(thickness):
                return np.dot(mean_air, np.exp(-mu_per_mm * thickness)) - threshold
            if difference(100.0) >= 0:
                return 100.0
            return float(optimize.brentq(difference, 0.0, 100.0, xtol=1e-12))

        t_limit = thickness_for_penalty(time_penalty_limit)
        # Include a full trade-off curve and the actual feasible boundary.
        t_three = thickness_for_penalty(3.0)
        t_plot = thickness_for_penalty(max(15.0, time_penalty_limit))
        thicknesses = np.unique(np.concatenate((
            np.linspace(0.0, t_three, 61), np.linspace(t_three, t_plot, 31), [t_limit]
        )))
        results.extend(evaluate(thickness) for thickness in thicknesses)

        candidates = [evaluate(0.0), evaluate(t_limit)]
        if t_limit > 0:
            scan = np.linspace(0.0, t_limit, 65)
            sampled = [evaluate(thickness) for thickness in scan]
            candidates.extend(sampled)
            for index in range(1, len(scan) - 1):
                value = sampled[index]['Dynamic_CNR']
                if (value >= sampled[index - 1]['Dynamic_CNR']
                        and value >= sampled[index + 1]['Dynamic_CNR']
                        and (value > sampled[index - 1]['Dynamic_CNR']
                             or value > sampled[index + 1]['Dynamic_CNR'])):
                    solution = optimize.minimize_scalar(
                        lambda thickness: -evaluate(thickness)['Dynamic_CNR'],
                        bounds=(scan[index - 1], scan[index + 1]), method='bounded',
                        options={'xatol': 1e-10},
                    )
                    if solution.success:
                        candidates.append(evaluate(solution.x))
        feasible = [row for row in candidates if row['Time_Penalty'] <= time_penalty_limit * (1 + 1e-10)]
        # Stable tie-breaking chooses no/thinner filter when CNR is identical.
        top_results.append(max(feasible, key=lambda row: (row['Dynamic_CNR'], -row['Thickness_mm'])))

    sweep = pd.DataFrame(results, columns=columns)
    optimum = pd.DataFrame(top_results, columns=columns).sort_values('Dynamic_CNR', ascending=False)
    for frame in (sweep, optimum):
        frame.attrs['detector_model'] = 'first_collision_deterministic_response'
        frame.attrs['noise_model'] = 'independent_A_plus_B_compound_Poisson'
        frame.attrs['cnr_units'] = '1/sqrt(keV detector capacity)'
    return sweep, optimum

import apply_filter_and_fluorescence as aff

def run_fluorescence_optimization(energies, raw_photons, mat1, thick1, candidate_materials=None, geom_params=None, downstream_transmission=None):
    """Rank second filters by fluorescence-energy suppression and primary retention.

    Count fluorescence induced by both the primary beam and upstream
    fluorescence. Scatter is excluded from this specific objective. Optional
    downstream_transmission weights the spectra after both filters (e.g. air
    attenuation to the target); it must not change the spectrum exciting them.
    """
    energies, raw_photons = validate_spectrum(energies, raw_photons)
    downstream = (np.ones_like(energies) if downstream_transmission is None
                  else np.asarray(downstream_transmission, dtype=float))
    if (downstream.shape != energies.shape or not np.all(np.isfinite(downstream))
            or np.any(downstream < 0) or np.any(downstream > 1)):
        raise ValueError('Downstream transmission must match the energy grid and lie between 0 and 1.')
    root = Path(__file__).resolve().parent
    att_df = pd.read_csv(root / 'attenuation_coefficients_long.csv')
    edpl_folder = root / 'pyne_edpl'
    
    geom_params = {'beam_angle_deg': 30.0} if geom_params is None else dict(geom_params)
    
    # 1. Base Layer (Filter 1)
    _, phi_fluor1, _, trans_factor1 = aff.process_single_layer(
        energies, raw_photons, mat1, thick1, att_df, edpl_folder, geom_params, calc_fluorescence=True, calc_scatter=False
    )
    phi_trans1 = raw_photons * trans_factor1
    
    # Base Power and Fluorescence Power
    P0 = integrate_spectrum(energies, phi_trans1 * energies * downstream)
    F0 = integrate_spectrum(energies, phi_fluor1 * energies * downstream)
    
    if P0 <= 0 or F0 <= 0:
        return pd.DataFrame(), pd.DataFrame()
        
    if candidate_materials is None:
        candidate_materials = ['Al', 'Cu', 'Mo', 'Ag', 'Sn', 'W', 'Pb']
    thicknesses = np.concatenate(([0.0], np.linspace(0.01, 2.0, 50)))
    
    results = []
    
    for mat2 in candidate_materials:
        for t2 in thicknesses:
            _, phi_fluor_from_main2, _, trans_factor2 = aff.process_single_layer(
                energies, phi_trans1, mat2, t2, att_df, edpl_folder, geom_params, calc_fluorescence=True, calc_scatter=False
            )
            phi_trans2 = phi_trans1 * trans_factor2
            
            # Upstream fluorescence can excite lower-Z fluorescence in layer 2.
            phi_fluor1_attenuated_total, _, _, _ = aff.process_single_layer(
                energies, phi_fluor1, mat2, t2, att_df, edpl_folder, geom_params, calc_fluorescence=True, calc_scatter=False
            )
            
            P = integrate_spectrum(energies, phi_trans2 * energies * downstream)
            total_F_spectrum = phi_fluor1_attenuated_total + phi_fluor_from_main2
            F = integrate_spectrum(energies, total_F_spectrum * energies * downstream)
            
            p_kept = P / P0 if P0 > 0 else 0
            f_elim = 1.0 - (F / F0) if F0 > 0 else 0
            # Negative elimination exposes filters that add fluorescence;
            # the zero-thickness candidate always offers an unchanged beam.
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
            
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(results)
    valid_df = df[df['P_kept'] > 0.01]
    if valid_df.empty:
        valid_df = df
        
    top_per_mat = valid_df.loc[valid_df.groupby('Filter_Material')['Score'].idxmax()].sort_values(by='Score', ascending=False)
    
    return df, top_per_mat
