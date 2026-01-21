
# Now compute fluorescence for both materials using EPDL mu/tau tables present in csv files in directory
filters = {
    'Mo': {'df': mo, 'thickness_mm': 0.10, 'density': 10.28, 'omega': omega_Mo, 'Eka': E_Mo_Kalpha, 'Ekb': E_Mo_Kbeta},
    'Pb': {'df': pb, 'thickness_mm': 0.025, 'density': 11.34, 'omega': omega_Pb, 'Eka': E_Pb_Kalpha, 'Ekb': E_Pb_Kbeta}
}
Kbeta_to_Kalpha = 0.12
BR_Kalpha = 1.0/(1.0 + Kbeta_to_Kalpha)
BR_Kbeta = 1.0 - BR_Kalpha

results = []
for name, info in filters.items():
    df = info['df']
    # ensure df sorted by E_keV
    df = df.sort_values('E_keV')
    mu_on_spec = np.interp(E_spec, df['E_keV'], df['mu_over_rho_cm2_per_g'])
    tau_on_spec = np.interp(E_spec, df['E_keV'], df['tauK_over_rho_cm2_per_g'])
    rho_t = info['density'] * (info['thickness_mm']/10.0)
    P_abs = 1.0 - np.exp(-mu_on_spec * rho_t)
    fracK = np.zeros_like(mu_on_spec)
    nz = mu_on_spec > 0
    fracK[nz] = tau_on_spec[nz] / mu_on_spec[nz]
    R_K = phi0 * P_abs * fracK
    Y_Kprod = R_K * info['omega']
    Y_Ktotal = np.trapz(Y_Kprod, E_spec)
    Y_Kalpha = Y_Ktotal * BR_Kalpha
    Y_Kbeta  = Y_Ktotal * BR_Kbeta
    # Escape probabilities at emission energies
    mu_Eka = np.interp(info['Eka'], df['E_keV'], df['mu_over_rho_cm2_per_g'])
    mu_Ekb = np.interp(info['Ekb'], df['E_keV'], df['mu_over_rho_cm2_per_g'])
    def Pesc(mu_val):
        denom = mu_val * rho_t
        if denom <= 0:
            return 1.0
        return (1.0 - np.exp(-denom))/denom
    Pesc_a = Pesc(mu_Eka); Pesc_b = Pesc(mu_Ekb)
    Phi_a_esc = Y_Kalpha * Pesc_a
    Phi_b_esc = Y_Kbeta  * Pesc_b
    eVJ = 1.602176634e-19
    Power_a = Phi_a_esc * info['Eka'] * 1e3 * eVJ
    Power_b = Phi_b_esc * info['Ekb'] * 1e3 * eVJ
    results.append({
        'material': name,
        'thickness_mm': info['thickness_mm'],
        'rho_g_cm3': info['density'],
        'omega_K': info['omega'],
        'BR_Kalpha': BR_Kalpha,
        'BR_Kbeta': BR_Kbeta,
        'Y_Ktotal_produced_photons_per_s': float(Y_Ktotal),
        'Y_Kalpha_produced_photons_per_s': float(Y_Kalpha),
        'Y_Kbeta_produced_photons_per_s': float(Y_Kbeta),
        'Phi_Kalpha_escaping_photons_per_s': float(Phi_a_esc),
        'Phi_Kbeta_escaping_photons_per_s': float(Phi_b_esc),
        'Pesc_Kalpha': float(Pesc_a),
        'Pesc_Kbeta': float(Pesc_b),
        'Power_Kalpha_W': float(Power_a),
        'Power_Kbeta_W': float(Power_b)
    })

res_df = pd.DataFrame(results)
res_path = Path('./fluorescence_results_summary_from_epdl.csv')
res_df.to_csv(res_path, index=False)
