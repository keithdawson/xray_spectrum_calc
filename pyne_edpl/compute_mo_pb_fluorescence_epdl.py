# save as compute_mo_pb_fluorescence_epdl.py and run with Python 3.10+
import io, os, math, sys, requests, zipfile
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- User inputs ----------
spectrum_csv = Path('./w_tungsten_225kv_spectrum.csv')  # path to existing tungsten spectrum CSV
out_dir = Path('./epdl_results'); out_dir.mkdir(exist_ok=True)
normalize_total_photons = 1.0e8   # same normalization we used earlier (scales linearly)
# filter thicknesses
t_Mo_mm = 0.10    # mm
t_Pb_mm = 0.025   # mm
# fluorescence params (these are standard recommended values)
omega_K_values = {'Mo': 0.765, 'Pb': 0.96}   # recommended values (Krause / tabulated compilations)
Kbeta_to_Kalpha_ratio = 0.12
BR_Kalpha = 1.0 / (1.0 + Kbeta_to_Kalpha_ratio)
BR_Kbeta  = 1.0 - BR_Kalpha
# emission line energies (keV)
E_Mo_Kalpha = 17.479372
E_Mo_Kbeta  = 19.608
E_Pb_Kalpha = 74.969
E_Pb_Kbeta  = 84.94

# ---------- URLs (LLNL EPDL element files & NIST XCOM) ----------
# LLNL EPDL files (text) — these are the 2025/EPDL.ELEMENTS endpoints discovered
EPDL_BASE = "https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL.ELEMENTS"
EPDL_URLS = {'Mo': f"{EPDL_BASE}/ZA042000", 'Pb': f"{EPDL_BASE}/ZA082000"}

# NIST XCOM element pages (we will use the downloadable table)
NIST_XCOM_BASE = "https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab"
NIST_XCOM_URLS = {'Mo': f"{NIST_XCOM_BASE}/z42.html", 'Pb': f"{NIST_XCOM_BASE}/z82.html"}

# ---------- helper: parse EPDL element file to extract MF=23,MT=534 (K subshell) and MF=23,MT=501 (total) ----------
def download_text(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def parse_epdl_ksubshell(epdl_text):
    # ENDF-like file; we will scan for the MF=23 blocks and MT=534 (K) and MT=522/501 totals.
    lines = epdl_text.splitlines()
    # Simple state machine: look for lines that start with an integer material id and then MF/MT tokens
    # In LLNL element file found on the site, MF/MT are embedded; a robust parser would parse columns.
    # We'll find the start of the "23/534" string (or "23 534") and then read the following numeric pairs until next header.
    txt = '\n'.join(lines)
    # find the MF/MT header for 23/534 and then capture the numeric block following until a blank line or next '23/' occurrence
    def extract_block(tag):
        idx = txt.find(tag)
        if idx < 0:
            return None
        start = idx + len(tag)
        # take substring after the tag; numeric data are space-separated floats; stop at two consecutive newlines or next '23/' token
        sub = txt[start:]
        # find next occurrence of '\n\n' or '23/' after start
        stop_candidates = []
        p1 = sub.find('\n\n')
        if p1>=0: stop_candidates.append(p1)
        p2 = sub.find('\n23/')
        if p2>=0: stop_candidates.append(p2)
        # also stop at '\n4200' which marks next major header often
        p3 = sub.find('\n4200')
        if p3>=0: stop_candidates.append(p3)
        stop = min(stop_candidates) if stop_candidates else None
        block = sub if stop is None else sub[:stop]
        # extract numbers from block
        nums = []
        for tok in block.replace('\n',' ').split():
            try:
                nums.append(float(tok))
            except:
                pass
        return nums

    # two tags that often appear: "23/534" and "23 534" - try both
    for tag in ("23/534", "23 534", "23/ 534"):
        nums = extract_block(tag)
        if nums:
            # EPDL subshell blocks normally list: number of points, then pairs of (E, sigma) or log-log encoded arrays.
            # We need to interpret pairs: find a plausible sequence of (E, sigma) within the numeric list
            # Heuristic: take pairs (nums[0], nums[1]), (nums[2], nums[3]), ...
            if len(nums) >= 4:
                pairs = [(nums[i], nums[i+1]) for i in range(0, (len(nums)//2)*2, 2)]
                return pairs
    return None

# ---------- helper: parse NIST XCOM element html tables for mu/rho ----------
def parse_nist_mu_rho(html_text):
    # NIST pages contain rows like "1.00000E-03  5.210E+03  5.197E+03"
    rows = []
    for line in html_text.splitlines():
        parts = line.strip().split()
        # require at least 3 numeric parts and exponent notation
        if len(parts) >= 3:
            try:
                e = float(parts[0])
                mu = float(parts[1])
                # mu is given in MeV units on NIST page; their table E is MeV — convert to keV by *1e3
                rows.append((e*1e3, mu))
            except:
                continue
    # remove duplicates and return sorted arrays (E_keV, mu_over_rho)
    if not rows:
        raise RuntimeError("no numeric table rows found in NIST HTML (page format changed?)")
    rows_sorted = sorted(set(rows))
    E = np.array([r[0] for r in rows_sorted])
    mu = np.array([r[1] for r in rows_sorted])
    return E, mu

# ---------- main pipeline ----------
def main():
    # load spectrum CSV
    if not spectrum_csv.exists():
        raise FileNotFoundError(f"Input spectrum not found: {spectrum_csv}")
    spec = pd.read_csv(spectrum_csv)
    Egrid_spec = spec['E_keV'].to_numpy()
    phi0 = spec['photons_per_keV'].to_numpy()

    results = []
    for sym in ('Mo', 'Pb'):
        print(f"[+] processing {sym}")

        # download EPDL element file
        epdl_text = download_text(EPDL_URLS[sym])
        # extract K-subshell pairs
        pairs = parse_epdl_ksubshell(epdl_text)
        if not pairs:
            print(f"  (!) could not extract 23/534 block for {sym}; aborting for this element")
            continue
        # build arrays
        E_sub = np.array([p[0] for p in pairs])   # energies (keV or eV?)  -- EPDL uses eV in some forms, be careful
        sigma_sub = np.array([p[1] for p in pairs])  # cross sections in ??? units (barns/atom?) EPDL uses barns/atom
        # NOTE: EPDL numeric units must be converted: EPDL energies are usually in eV -> convert to keV if needed.
        # Heuristics: if max(E_sub) > 1e3 assume eV; convert eV->keV dividing by 1e3
        if E_sub.max() > 1e3:
            E_sub = E_sub / 1e3

        # If sigma units are barns/atom, convert to mass attenuation μ/ρ by:
        # mu_over_rho (cm^2/g) = (sigma_barn * 1e-24 cm^2/barn) * Na / A
        # where Na = Avogadro's number, A = atomic mass (g/mol) of element
        # We'll need atomic masses:
        atomic_mass = {'Mo': 95.95, 'Pb': 207.2}[sym]
        NA = 6.02214076e23
        # EPDL sigma may represent photoionization cross section (barns/atom)
        mu_sub = sigma_sub * 1e-24 * NA / atomic_mass  # approximate per mass (cm^2/g)

        # Now get total mu/rho from NIST XCOM page
        nist_html = download_text(NIST_XCOM_URLS[sym])
        E_nist, mu_nist = parse_nist_mu_rho(nist_html)   # E in keV, mu in cm^2/g
        # Interpolate totals onto the spectrum grid and onto E_sub if needed
        mu_tot_on_spec = np.interp(Egrid_spec, E_nist, mu_nist, left=mu_nist[0], right=mu_nist[-1])
        tauK_on_spec = np.interp(Egrid_spec, E_sub, mu_sub, left=0.0, right=0.0)

        # convert thickness
        if sym == 'Mo':
            rho = 10.28; t_cm = t_Mo_mm/10.0
            omega_K = omega_K_values['Mo']
            E_Kalpha, E_Kbeta = E_Mo_Kalpha, E_Mo_Kbeta
        else:
            rho = 11.34; t_cm = t_Pb_mm/10.0
            omega_K = omega_K_values['Pb']
            E_Kalpha, E_Kbeta = E_Pb_Kalpha, E_Pb_Kbeta

        rho_t = rho * t_cm

        # compute P_abs, fraction of absorptions producing K-vacancy
        P_abs = 1.0 - np.exp(-mu_tot_on_spec * rho_t)
        frac_K = np.zeros_like(mu_tot_on_spec)
        nz = mu_tot_on_spec > 0
        frac_K[nz] = tauK_on_spec[nz] / mu_tot_on_spec[nz]
        R_K = phi0 * P_abs * frac_K
        Y_Kprod = R_K * omega_K
        Y_Ktotal = np.trapz(Y_Kprod, Egrid_spec)
        Y_Kalpha = Y_Ktotal * BR_Kalpha
        Y_Kbeta  = Y_Ktotal * BR_Kbeta

        # escape probabilities (use mu at emission energies from NIST)
        mu_Kalpha = np.interp(E_Kalpha, E_nist, mu_nist)
        mu_Kbeta  = np.interp(E_Kbeta,  E_nist, mu_nist)
        def Pesc(mu_val):
            denom = mu_val * rho_t
            if denom == 0: return 1.0
            return (1.0 - np.exp(-denom)) / denom
        Pesc_alpha = Pesc(mu_Kalpha)
        Pesc_beta  = Pesc(mu_Kbeta)
        Phi_alpha_esc = Y_Kalpha * Pesc_alpha
        Phi_beta_esc  = Y_Kbeta * Pesc_beta

        # power
        eV_to_J = 1.602176634e-19
        Power_alpha = Phi_alpha_esc * E_Kalpha * 1e3 * eV_to_J
        Power_beta  = Phi_beta_esc * E_Kbeta * 1e3 * eV_to_J

        # save mu & tau tables interpolated to common grid (spectrum grid)
        df_mu_tau = pd.DataFrame({
            'E_keV': Egrid_spec,
            'mu_over_rho_cm2_per_g': mu_tot_on_spec,
            'tauK_over_rho_cm2_per_g': tauK_on_spec
        })
        df_mu_tau.to_csv(out_dir / f"{sym.lower()}_mu_tau_from_epdl.csv", index=False)

        results.append({
            'material': sym,
            'thickness_mm': (t_Mo_mm if sym=='Mo' else t_Pb_mm),
            'rho_g_cm3': rho,
            'Y_K_total_produced_photons_per_s': float(Y_Ktotal * (normalize_total_photons / np.sum(phi0))),   # scale to normalization
            'Y_Kalpha_produced_photons_per_s': float(Y_Kalpha * (normalize_total_photons / np.sum(phi0))),
            'Y_Kbeta_produced_photons_per_s': float(Y_Kbeta  * (normalize_total_photons / np.sum(phi0))),
            'Phi_Kalpha_escaping_photons_per_s': float(Phi_alpha_esc * (normalize_total_photons / np.sum(phi0))),
            'Phi_Kbeta_escaping_photons_per_s': float(Phi_beta_esc  * (normalize_total_photons / np.sum(phi0))),
            'Power_Kalpha_W': float(Power_alpha * (normalize_total_photons / np.sum(phi0))),
            'Power_Kbeta_W': float(Power_beta * (normalize_total_photons / np.sum(phi0)))
        })

    # write results
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "fluorescence_results_summary_epdl.csv", index=False)
    print("Wrote results to:", out_dir)
    return out_dir

if __name__ == '__main__':
    main()
