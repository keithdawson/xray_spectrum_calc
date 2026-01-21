import pandas as pd
import numpy as np
import glob
import os
import re

def generate_master_csv():
    # Folder containing the individual z##.csv files
    input_folder = 'extract_atten_coeff'
    output_file = 'attenuation_coefficients_long.csv'
    
    # Check if folder exists
    if not os.path.exists(input_folder):
        # Fallback to current directory if folder not found
        if glob.glob('z*.csv'):
            input_folder = '.'
        else:
            print(f"Error: Folder '{input_folder}' not found and no z*.csv files in current dir.")
            return

    # Define energy grid
    # Range: 1 keV to 500 keV (0.001 to 0.5 MeV)
    # Note: Adjusted max energy to 0.5 MeV to cover the 225 kVp spectrum range.
    # Use integer steps (keV) to avoid floating point accumulation errors
    min_kev = 1
    max_kev = 500
    
    # Create grid in keV then convert to MeV
    kev_grid = np.arange(min_kev, max_kev + 1)
    energy_grid = kev_grid.astype(float) / 1000.0
    
    # Find all CSV files matching z<number>.csv
    csv_files = glob.glob(os.path.join(input_folder, 'z*.csv'))
    
    all_data = []
    
    print(f"Found {len(csv_files)} CSV files in '{input_folder}'. Processing...")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # Extract Atomic Number (Z) from filename
        match = re.match(r'z(\d+)\.csv', filename, re.IGNORECASE)
        if not match:
            continue
            
        z = int(match.group(1))
        
        try:
            # Read the individual CSV
            df = pd.read_csv(csv_file)
            
            # Normalize column names to handle potential case variations
            df.columns = [c.strip() for c in df.columns]
            
            # Get data arrays (assuming standard column names from previous script)
            x = df['Energy_MeV'].values
            y_mu = df['Mu_over_rho'].values
            y_mu_en = df['mu_en_over_rho'].values
            
            # Interpolate onto the master energy grid
            new_mu = np.interp(energy_grid, x, y_mu)
            new_mu_en = np.interp(energy_grid, x, y_mu_en)
            
            # Round to remove floating point artifacts (drop insignificant digits)
            new_mu = np.round(new_mu, 8)
            new_mu_en = np.round(new_mu_en, 8)
            
            # Create DataFrame for this element
            element_df = pd.DataFrame({
                'Energy_MeV': energy_grid,
                'Element': z,
                'Mu_over_rho': new_mu,
                'Mu_en_over_rho': new_mu_en
            })
            
            all_data.append(element_df)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_data:
        print("No valid data found.")
        return

    # Combine all elements
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by Element, then Energy
    master_df = master_df.sort_values(by=['Element', 'Energy_MeV'])
    
    # Format Element column to be integer
    master_df['Element'] = master_df['Element'].astype(int)
    
    # Save to CSV
    output_path = os.path.join(input_folder, output_file)
    master_df.to_csv(output_path, index=False)
    
    print(f"Successfully generated {output_path} with {len(master_df)} rows.")

if __name__ == "__main__":
    generate_master_csv()