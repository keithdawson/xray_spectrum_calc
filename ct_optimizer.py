"""Primary-ray CNR and fluorescence-suppression filter searches."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from ct_physics import attenuation_exponent, detector_response, validate_spectrum
from spectrum_utils import energy_bin_widths, integrate_spectrum


def run_filter_optimization(energies, raw_photons, target_a_mat, target_a_thick, target_b_mat, target_b_thick, shadow1_mat, shadow1_thick, shadow2_mat, shadow2_thick, scint_mat, scint_thick, path_multiplier=1.0, time_penalty_limit=5.0, extra_filters=None, filter_materials=None, include_secondary_fog=False, geom_params=None):
    """Maximize idealized CNR at fixed air-path detector energy capacity.

    Let S be deposited-energy rate and V be its compound-Poisson variance
    rate. With air-path capacity Q, exposure is Q/S_air, and the CNR of
    independent A/B measurements is sqrt(Q)*abs(S_A-S_B)/sqrt(S_air*(V_A+V_B)).
    Dynamic_CNR returns this coefficient per sqrt(keV of capacity), not an
    absolute CNR. Time_Penalty is S_air(unfiltered)/S_air_primary(filtered).

    When ``include_secondary_fog=False`` (default), the response matches
    calculate_contrast and excludes filter secondaries. When
    ``include_secondary_fog=True``, spatially broad K-fluorescence and
    Compton/Rayleigh scatter emitted by the candidate filter within the
    acceptance cone are added as unmodulated background to the air-path
    detector load ``S_air`` and object-attenuated measurement variances
    ``V_A`` and ``V_B``.
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
    mean_weight = mean_kernel * widths
    variance_weight = variance_kernel * widths
    mean_air = raw_photons * mean_weight
    variance_air = raw_photons * variance_weight
    unfiltered_s_max = np.sum(mean_air)
    if unfiltered_s_max <= 0:
        raise ValueError('Unfiltered detector signal is zero. Check spectrum and scintillator thickness.')

    use_secondary_fog = bool(include_secondary_fog)
    if use_secondary_fog:
        root = Path(__file__).resolve().parent
        att_df = _attenuation_table()
        edpl_folder = root / 'pyne_edpl'
        active_geom = {'beam_angle_deg': 30.0} if geom_params is None else dict(geom_params)

    materials = list(filter_materials) if filter_materials is not None else ['Al', 'Fe', 'Cu', 'Mo', 'Ag', 'Sn', 'Ta', 'W', 'Pb']
    for material in extra_filters or []:
        if material not in materials:
            materials.append(material)
    materials = list(dict.fromkeys(materials))
    columns = ['Filter_Material', 'Thickness_mm', 'Signal_Max', 'Signal_A', 'Signal_B',
               'Variance_A', 'Variance_B', 'Dynamic_CNR', 'Time_Penalty', 'Secondary_Fog_Pct']
    results, top_results = [], []

    for material in materials:
        mu_per_mm = attenuation_exponent(material, energies, 1.0)
        fog_cache = {0.0: (0.0, 0.0, 0.0)}

        def eval_fog_exact(thickness):
            t_key = float(thickness)
            if t_key <= 0.0:
                return 0.0, 0.0, 0.0
            if t_key in fog_cache:
                return fog_cache[t_key]
            _, phi_fluor, phi_scatter, _ = aff.process_single_layer(
                energies, raw_photons, material, t_key, att_df, edpl_folder,
                active_geom, calc_fluorescence=True, calc_scatter=True
            )
            phi_sec = phi_fluor + phi_scatter
            s_fog = float(np.dot(phi_sec, mean_weight))
            fog_var_spectrum = phi_sec * variance_weight
            v_a_fog = float(np.dot(fog_var_spectrum, path_a))
            v_b_fog = float(np.dot(fog_var_spectrum, path_b))
            res_tuple = (s_fog, v_a_fog, v_b_fog)
            fog_cache[t_key] = res_tuple
            return res_tuple

        def thickness_for_penalty(penalty):
            if penalty == 1.0:
                return 0.0
            # Find the boundary using primary signal instead of an overflowing ratio.
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

        fog_interp_t = None
        if use_secondary_fog and len(energies) > 32 and max(t_plot, t_limit) > 0:
            fog_interp_t = np.unique(np.concatenate((
                [0.0, t_three, t_limit],
                np.linspace(0.0, max(t_plot, t_limit), 19),
            )))
            fog_samples = np.array([eval_fog_exact(t) for t in fog_interp_t], dtype=float)
            fog_s_grid = fog_samples[:, 0]
            fog_va_grid = fog_samples[:, 1]
            fog_vb_grid = fog_samples[:, 2]

        def evaluate(thickness, exact_fog=False):
            t_val = float(thickness)
            transmission = np.exp(-mu_per_mm * t_val)
            filtered_mean = mean_air * transmission
            filtered_variance = variance_air * transmission
            s_air_prim = float(np.sum(filtered_mean))
            s_a = float(np.dot(filtered_mean, path_a))
            s_b = float(np.dot(filtered_mean, path_b))
            v_a = float(np.dot(filtered_variance, path_a))
            v_b = float(np.dot(filtered_variance, path_b))
            if use_secondary_fog and t_val > 0.0:
                if exact_fog or fog_interp_t is None or t_val in fog_cache:
                    s_fog, va_fog, vb_fog = eval_fog_exact(t_val)
                else:
                    s_fog = float(np.interp(t_val, fog_interp_t, fog_s_grid))
                    va_fog = float(np.interp(t_val, fog_interp_t, fog_va_grid))
                    vb_fog = float(np.interp(t_val, fog_interp_t, fog_vb_grid))
            else:
                s_fog, va_fog, vb_fog = 0.0, 0.0, 0.0
            s_air = s_air_prim + s_fog
            v_a_tot = v_a + va_fog
            v_b_tot = v_b + vb_fog
            time_penalty = unfiltered_s_max / s_air_prim if s_air_prim > 0 else np.inf
            denominator = np.sqrt(v_a_tot + v_b_tot) * np.sqrt(s_air)
            cnr = abs(s_a - s_b) / denominator if denominator > 0 else 0.0
            fog_pct = (100.0 * s_fog / s_air) if s_air > 0 else 0.0
            return {'Filter_Material': material, 'Thickness_mm': t_val,
                    'Signal_Max': s_air, 'Signal_A': s_a, 'Signal_B': s_b,
                    'Variance_A': v_a_tot, 'Variance_B': v_b_tot,
                    'Dynamic_CNR': cnr, 'Time_Penalty': time_penalty,
                    'Secondary_Fog_Pct': fog_pct}

        thicknesses = np.unique(np.concatenate((
            np.linspace(0.0, t_three, 61), np.linspace(t_three, t_plot, 31), [t_limit]
        )))
        results.extend(evaluate(thickness) for thickness in thicknesses)

        candidates = [evaluate(0.0, exact_fog=True), evaluate(t_limit, exact_fog=True)]
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
                        candidates.append(evaluate(solution.x, exact_fog=True))
        feasible = [row for row in candidates if row['Time_Penalty'] <= time_penalty_limit * (1 + 1e-10)]
        # Stable tie-breaking chooses no/thinner filter when CNR is identical.
        best_row = max(feasible, key=lambda row: (row['Dynamic_CNR'], -row['Thickness_mm']))
        if use_secondary_fog:
            best_row = evaluate(best_row['Thickness_mm'], exact_fog=True)
        top_results.append(best_row)

    sweep = pd.DataFrame(results, columns=columns)
    optimum = pd.DataFrame(top_results, columns=columns).sort_values('Dynamic_CNR', ascending=False)
    for frame in (sweep, optimum):
        frame.attrs['detector_model'] = 'first_collision_deterministic_response'
        frame.attrs['noise_model'] = (
            'independent_A_plus_B_compound_Poisson_with_filter_secondary_fog'
            if use_secondary_fog else 'independent_A_plus_B_compound_Poisson'
        )
        frame.attrs['cnr_units'] = '1/sqrt(keV detector capacity)'
    return sweep, optimum

import apply_filter_and_fluorescence as aff
from ct_physics import _attenuation_table


def run_fluorescence_optimization(energies, raw_photons, mat1=None, thick1=None, candidate_materials=None, geom_params=None, downstream_transmission=None, include_scatter=False, upstream_filters=None):
    """Rank downstream filters by secondary-energy suppression and primary retention.

    Count fluorescence induced by both the primary beam and upstream
    fluorescence across one or more upstream filter layers (specified via
    ``(mat1, thick1)`` or ``upstream_filters=[{'material': ..., 'thickness': ...}, ...]``).
    When ``include_scatter=False`` (default), scatter is excluded
    and ``Score`` uses ``F_elim * P_kept**2``. When ``include_scatter=True``,
    Compton scatter from all layers is included and ``Score`` uses
    ``Secondary_elim * P_kept**2`` (preventing thick low-Z filters that replace
    K-fluorescence with heavier Compton scatter fog from ranking artificially
    high). Optional ``downstream_transmission`` weights the spectra after the
    filter stack (e.g. air attenuation to the target); it must not change the
    spectrum exciting the filters.
    """
    energies, raw_photons = validate_spectrum(energies, raw_photons)
    downstream = (np.ones_like(energies) if downstream_transmission is None
                  else np.asarray(downstream_transmission, dtype=float))
    if (downstream.shape != energies.shape or not np.all(np.isfinite(downstream))
            or np.any(downstream < 0) or np.any(downstream > 1)):
        raise ValueError('Downstream transmission must match the energy grid and lie between 0 and 1.')
    if upstream_filters is None:
        if mat1 is None or thick1 is None:
            raise ValueError('Specify either (mat1, thick1) or a non-empty upstream_filters sequence.')
        upstream_layers = [{'material': str(mat1), 'thickness': float(thick1)}]
    else:
        upstream_layers = [
            {'material': str(layer['material']), 'thickness': float(layer['thickness'])}
            for layer in upstream_filters
        ]
        if not upstream_layers:
            raise ValueError('upstream_filters must contain at least one filter layer.')
    root = Path(__file__).resolve().parent
    att_df = _attenuation_table()
    edpl_folder = root / 'pyne_edpl'
    
    geom_params = {'beam_angle_deg': 30.0} if geom_params is None else dict(geom_params)
    calc_scatter = bool(include_scatter)
    
    # 1. Propagate through Upstream Filter Stack (Layer 1 .. N)
    first_layer = upstream_layers[0]
    _, phi_fluor1, phi_scatter1, trans_factor1 = aff.process_single_layer(
        energies, raw_photons, first_layer['material'], first_layer['thickness'],
        att_df, edpl_folder, geom_params,
        calc_fluorescence=True, calc_scatter=calc_scatter
    )
    phi_trans1 = raw_photons * trans_factor1
    for layer_k in upstream_layers[1:]:
        _, phi_fluor_from_main, phi_scatter_from_main, trans_factor_k = aff.process_single_layer(
            energies, phi_trans1, layer_k['material'], layer_k['thickness'],
            att_df, edpl_folder, geom_params,
            calc_fluorescence=True, calc_scatter=calc_scatter
        )
        phi_fluor_out, _, phi_scatter_from_fluor, _ = aff.process_single_layer(
            energies, phi_fluor1, layer_k['material'], layer_k['thickness'],
            att_df, edpl_folder, geom_params,
            calc_fluorescence=True, calc_scatter=calc_scatter
        )
        phi_trans1 = phi_trans1 * trans_factor_k
        phi_fluor1 = (phi_fluor_out - phi_scatter_from_fluor) + phi_fluor_from_main
        if calc_scatter:
            phi_scatter1 = phi_scatter1 * trans_factor_k + phi_scatter_from_main + phi_scatter_from_fluor
    
    # Base Power, Fluorescence Power, and Scatter Power exiting the upstream stack
    P0 = integrate_spectrum(energies, phi_trans1 * energies * downstream)
    F0 = integrate_spectrum(energies, phi_fluor1 * energies * downstream)
    S0 = integrate_spectrum(energies, phi_scatter1 * energies * downstream) if calc_scatter else 0.0
    Sec0 = F0 + S0
    
    if P0 <= 0 or (Sec0 <= 0 if calc_scatter else F0 <= 0):
        return pd.DataFrame(), pd.DataFrame()
        
    if candidate_materials is None:
        candidate_materials = ['Al', 'Cu', 'Mo', 'Ag', 'Sn', 'W', 'Pb']
    thicknesses = np.concatenate(([0.0], np.linspace(0.01, 2.0, 50)))
    
    results = []
    
    for mat2 in candidate_materials:
        for t2 in thicknesses:
            _, phi_fluor_from_main2, phi_scatter_from_main2, trans_factor2 = aff.process_single_layer(
                energies, phi_trans1, mat2, t2, att_df, edpl_folder, geom_params,
                calc_fluorescence=True, calc_scatter=calc_scatter
            )
            phi_trans2 = phi_trans1 * trans_factor2
            
            # Upstream fluorescence can excite lower-Z fluorescence in layer 2.
            phi_fluor1_out, _, phi_scatter_from_fluor1, _ = aff.process_single_layer(
                energies, phi_fluor1, mat2, t2, att_df, edpl_folder, geom_params,
                calc_fluorescence=True, calc_scatter=calc_scatter
            )
            phi_fluor1_attenuated_total = phi_fluor1_out - phi_scatter_from_fluor1
            
            P = integrate_spectrum(energies, phi_trans2 * energies * downstream)
            total_F_spectrum = phi_fluor1_attenuated_total + phi_fluor_from_main2
            F = integrate_spectrum(energies, total_F_spectrum * energies * downstream)
            if calc_scatter:
                total_S_spectrum = (
                    phi_scatter1 * trans_factor2 + phi_scatter_from_main2 + phi_scatter_from_fluor1
                )
                S = integrate_spectrum(energies, total_S_spectrum * energies * downstream)
            else:
                S = 0.0
            Sec = F + S
            
            p_kept = P / P0 if P0 > 0 else 0.0
            f_elim = 1.0 - (F / F0) if F0 > 0 else 0.0
            sec_elim = 1.0 - (Sec / Sec0) if Sec0 > 0 else 0.0
            # Negative elimination exposes filters that add fluorescence/scatter;
            # the zero-thickness candidate always offers an unchanged beam.
            active_elim = sec_elim if calc_scatter else f_elim
            score = active_elim * (p_kept ** 2)
            
            results.append({
                'Filter_Material': mat2,
                'Thickness_mm': t2,
                'P_kept': p_kept,
                'F_elim': f_elim,
                'Secondary_elim': sec_elim,
                'Score': score,
                'P_Total': P,
                'F_Total': F,
                'S_Total': S,
                'Secondary_Total': Sec,
            })
            
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(results)
    valid_df = df[df['P_kept'] > 0.01]
    if valid_df.empty:
        valid_df = df
        
    top_per_mat = valid_df.loc[valid_df.groupby('Filter_Material')['Score'].idxmax()].sort_values(by='Score', ascending=False)
    
    return df, top_per_mat
