#!/usr/bin/env python3
"""
extract_epdl_with_pyne.py

Parses EPDL (ZA*.txt) files with PyNE's endf.Evaluation and extracts
photon interaction cross sections (total and subshells).

Outputs:
  ZAxxxxx_extracted.csv            # energy grid (1–450 keV, 1 keV step)
  ZAxxxxx_extracted_metadata.csv   # MT mapping, binding energies, yields

Requires:
  pip install pyne
"""

import os
import glob
import math
import numpy as np
import pandas as pd
from pathlib import Path
from pyne.endf import Evaluation
from pyne import data as pyne_data

# constants
NA = 6.02214076e23
E_GRID_KEV = np.arange(1.0, 451.0, 1.0)  # 1–450 keV

def barns_per_atom_to_mu_rho(sigma_barns, atomic_mass_g_mol):
    """Convert barns/atom → cm^2/g."""
    return np.asarray(sigma_barns, dtype=float) * 1e-24 * NA / float(atomic_mass_g_mol)

def safe_get_xs_arrays(reaction):
    """Return (energies_keV, sigma_barns) from reaction.xs if possible."""
    xs = getattr(reaction, "xs", None)
    if xs is None:
        return None, None

    e = None
    sig = None

    if hasattr(xs, "e_int") and hasattr(xs, "xs"):
        e = np.asarray(xs.e_int)
        sig = np.asarray(xs.xs)
    elif hasattr(xs, "energies") and hasattr(xs, "values"):
        e = np.asarray(xs.energies)
        sig = np.asarray(xs.values)
    elif isinstance(xs, dict):
        e = np.asarray(xs.get("e_int") or xs.get("energies"))
        sig = np.asarray(xs.get("xs") or xs.get("sigma"))

    if e is None or sig is None:
        return None, None

    # convert eV → keV if needed
    if e.max() > 1e3:
        e = e / 1e3

    return e, sig

def extract_file(path):
    """Extract photon interaction xs from a ZA*.txt file."""
    path = Path(path)
    base = path.stem
    print(f"[+] Processing {path.name}")

    ev = Evaluation(str(path))
    ev.read(reactions=None)

    # try to get atomic mass
    atomic_mass = None
    try:
        Z = int(base[2:5])  # parse ZA*** prefix
        symbol = pyne_data.znum_to_symbol(Z)
        atomic_mass = pyne_data.atomic_mass(symbol)
    except Exception:
        atomic_mass = None
    if atomic_mass is None:
        raise RuntimeError(f"Could not determine atomic mass for {path.name}")

    components = {}
    meta_rows = []

    for mt, rxn in ev.reactions.items():
        e_keV, sig_barns = safe_get_xs_arrays(rxn)
        if e_keV is None:
            continue

        mu_rho = barns_per_atom_to_mu_rho(sig_barns, atomic_mass)

        bind_eV = getattr(rxn, "subshell_binding_energy", None)
        fluor_yield = getattr(rxn, "fluorescence_yield", None)

        if mt == 501:
            label = "total_interaction"
        elif mt == 522:
            label = "photoelectric_total"
        elif 534 <= mt <= 550:
            # subshells
            shell = rxn.__class__.__name__ if hasattr(rxn, "__class__") else "subshell"
            bind_keV = bind_eV / 1e3 if bind_eV else None
            if bind_keV:
                label = f"subshell_MT{mt}_{bind_keV:.2f}keV"
            else:
                label = f"subshell_MT{mt}"
        else:
            label = f"MT{mt}"

        components[label] = (e_keV, mu_rho)
        meta_rows.append({
            "MT": mt,
            "Label": label,
            "Binding_eV": float(bind_eV) if bind_eV else None,
            "Fluorescence_yield": float(fluor_yield) if fluor_yield else None,
            "Points": len(e_keV)
        })

    if not components:
        raise RuntimeError(f"No photon cross sections found in {path.name}")

    # interpolate onto common grid
    out = pd.DataFrame({"E_keV": E_GRID_KEV})
    for label, (e_src, mu_src) in components.items():
        out[label] = np.interp(E_GRID_KEV, e_src, mu_src, left=0.0, right=0.0)

    # choose one "main" total column
    if "total_interaction" in out.columns:
        out = out.rename(columns={"total_interaction": "mu_over_rho_cm2_per_g"})
    elif "photoelectric_total" in out.columns:
        out = out.rename(columns={"photoelectric_total": "mu_over_rho_cm2_per_g"})

    # save
    out_csv = path.with_name(f"{base}_extracted.csv")
    meta_csv = path.with_name(f"{base}_extracted_metadata.csv")
    out.to_csv(out_csv, index=False)
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)

    print(f"  wrote {out_csv} and {meta_csv}")

def main():
    files = sorted(glob.glob("ZA*.txt"))
    if not files:
        print("No ZA*.txt files found here.")
        return
    for f in files:
        try:
            extract_file(f)
        except Exception as e:
            print(f"ERROR {f}: {e}")

if __name__ == "__main__":
    main()
