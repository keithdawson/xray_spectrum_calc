import os
import glob
import pandas as pd
import re

MASS_ATTEN = pd.read_csv("mass_atten.csv")

csv_files = glob.glob("pyne_edpl/*_extracted.csv")
fixed_count = 0

for file in csv_files:
    if 'metadata' in file:
        continue
    
    base = os.path.basename(file)
    m = re.match(r"ZA0*([0-9]+)", base)
    if not m:
        continue
        
    Z = int(m.group(1))
    
    row = MASS_ATTEN[MASS_ATTEN["Z"] == Z]
    if row.empty:
        print(f"Skipping Z={Z}, no mass_atten data")
        continue
        
    z_over_a = float(row["Z/A"].values[0])
    correction_factor = z_over_a ** 2
    
    df = pd.read_csv(file)
    
    # We must only multiply the cross section columns, NOT E_keV!
    for col in df.columns:
        if col != 'E_keV':
            df[col] = df[col] * correction_factor
            
    df.to_csv(file, index=False)
    fixed_count += 1

print(f"Successfully fixed {fixed_count} PYNE extracted files with correction factor (Z/A)^2.")
