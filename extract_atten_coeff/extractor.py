import re
import pandas as pd
import numpy as np
import glob
import os

def extract_data_from_file(filename):
    """Extract (energy, mu/rho, mu_en/rho) from a NIST HTML file."""
    with open(filename, "r") as f:
        text = f.read()
    
    # ASCII block inside <PRE>
    ascii_block = re.search(r"<PRE>(.*?)</PRE>", text, re.S).group(1)
    
    # Regex for data lines
    pattern = re.compile(r"^\s*(?:[A-Za-z0-9]*)?\s*([0-9.E+-]+)\s+([0-9.E+-]+)\s+([0-9.E+-]+)", re.M)
    
    energies, mu_rho, mu_en_rho = [], [], []
    for match in pattern.finditer(ascii_block):
        energies.append(float(match.group(1)))
        mu_rho.append(float(match.group(2)))
        mu_en_rho.append(float(match.group(3)))
    
    return pd.DataFrame({
        "Energy_MeV": energies,
        "Mu_over_rho": mu_rho,
        "Mu_en_over_rho": mu_en_rho
    })

def interpolate_data(df, energy_grid):
    """Interpolate attenuation coefficients onto a uniform energy grid."""
    mu_interp = np.interp(energy_grid, df["Energy_MeV"], df["Mu_over_rho"])
    mu_en_interp = np.interp(energy_grid, df["Energy_MeV"], df["Mu_en_over_rho"])
    
    return pd.DataFrame({
        "Energy_MeV": energy_grid,
        "Mu_over_rho": mu_interp,
        "Mu_en_over_rho": mu_en_interp
    })

# Define energy grid (0.001 to 0.455 MeV in 0.001 steps)
energy_grid = np.arange(0.001, 0.455, 0.001)

# Process all HTML files in folder
all_files = glob.glob("*.html")  # adjust path if needed
long_format_list = []

for file in all_files:
    df_raw = extract_data_from_file(file)
    df_interp = interpolate_data(df_raw, energy_grid)
    
    # Use filename (strip "z" if present, keep atomic number)
    element_name = os.path.splitext(os.path.basename(file))[0]
    if element_name.lower().startswith("z"):
        element_name = element_name[1:]
    
    df_interp["Element"] = element_name
    long_format_list.append(df_interp)

# Combine all into one long table
long_df = pd.concat(long_format_list, ignore_index=True)

# Reorder columns
long_df = long_df[["Energy_MeV", "Element", "Mu_over_rho", "Mu_en_over_rho"]]

# Save combined CSV
long_df.to_csv("attenuation_coefficients_long.csv", index=False)

print("Saved attenuation_coefficients_long.csv with rows shaped like:")
print(long_df.head())
