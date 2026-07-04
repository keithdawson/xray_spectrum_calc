import os
from pathlib import Path
import pandas as pd
import numpy as np

edpl_folder = 'pyne_edpl'
elements = {'Al': 13, 'Fe': 26, 'Cu': 29, 'Mo': 42, 'Ag': 47, 'Sn': 50, 'W': 74, 'Au': 79, 'Pb': 82}

print(f"Checking EPDL Directory: '{edpl_folder}'\n")

if not os.path.exists(edpl_folder):
    print(f"ERROR: Directory '{edpl_folder}' does NOT exist!")
else:
    for sym, z in elements.items():
        p = Path(edpl_folder)
        candidates = list(p.glob(f"*ZA{z:03d}*.csv"))
        
        # Filter out metadata files
        data_files = [f for f in candidates if 'metadata' not in f.name]
        
        if not data_files:
            print(f"[{sym}] Z={z:03d}: NO DATA FILE FOUND.")
            continue
            
        file_path = data_files[0]
        try:
            df = pd.read_csv(file_path)
            # Check columns
            tau_col = None
            if 'tauK_over_rho_cm2_per_g' in df.columns:
                tau_col = 'tauK_over_rho_cm2_per_g'
            else:
                for c in df.columns:
                    if c.startswith('subshell_MT534'):
                        tau_col = c
                        break
            
            if 'E_keV' in df.columns and tau_col:
                print(f"[{sym}] Z={z:03d}: OK -> Found '{tau_col}' with {len(df)} rows.")
            else:
                print(f"[{sym}] Z={z:03d}: ERROR -> Missing required columns in {file_path.name}")
                
        except Exception as e:
            print(f"[{sym}] Z={z:03d}: ERROR -> Could not read {file_path.name}: {e}")
