import os
import pandas as pd
import numpy as np

def merge_attenuation():
    master_file = "attenuation_coefficients_long.csv"
    
    # 1. Regenerate NIST data using the original script
    # Let's temporarily call the original function
    import sys
    sys.path.append(os.path.abspath("extract_atten_coeff"))
    from generate_master_attenuation import generate_master_csv
    
    original_cwd = os.getcwd()
    os.chdir("extract_atten_coeff")
    generate_master_csv()
    os.chdir(original_cwd)
    
    nist_csv_path = "extract_atten_coeff/attenuation_coefficients_long.csv"
    nist_df = pd.read_csv(nist_csv_path)
    
    existing_elements = set(nist_df['Element'].unique())
    
    # Grid used by NIST
    energy_grid_kev = np.arange(1.0, 501.0, 1.0)
    
    all_data = [nist_df]
    
    pyne_elements_added = []
    
    for z in range(1, 87):
        if z in existing_elements:
            continue
            
        pyne_csv = f"pyne_edpl/ZA{z:03d}_extracted.csv"
        if not os.path.exists(pyne_csv):
            continue
            
        df = pd.read_csv(pyne_csv)
        
        mu = np.interp(energy_grid_kev, df['E_keV'], df['mu_over_rho_cm2_per_g'])
        mu = np.round(mu, 8)
        
        element_df = pd.DataFrame({
            'Energy_MeV': energy_grid_kev / 1000.0,
            'Element': z,
            'Mu_over_rho': mu,
            'Mu_en_over_rho': mu  # Dummy
        })
        
        all_data.append(element_df)
        pyne_elements_added.append(z)
        
    master_df = pd.concat(all_data, ignore_index=True)
    
    if 'Energy_KeV' not in master_df.columns:
        master_df['Energy_KeV'] = master_df['Energy_MeV'] * 1000.0
        
    master_df = master_df[['Energy_MeV', 'Energy_KeV', 'Element', 'Mu_over_rho', 'Mu_en_over_rho']]
        
    master_df = master_df.sort_values(by=['Element', 'Energy_KeV'])
    master_df.to_csv(master_file, index=False)

if __name__ == "__main__":
    merge_attenuation()
