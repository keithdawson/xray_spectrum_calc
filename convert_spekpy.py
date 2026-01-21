import pandas as pd
import numpy as np

def extrapolate_interp(x_new, x, y):
    """
    Linear interpolation with linear extrapolation at boundaries.
    """
    # Use numpy for interpolation (constant extrapolation by default)
    y_new = np.interp(x_new, x, y)
    
    # Extrapolate left
    if x_new[0] < x[0]:
        # Calculate slope between first two points
        slope_left = (y[1] - y[0]) / (x[1] - x[0])
        # Apply slope to points to the left of x[0]
        mask_left = x_new < x[0]
        y_new[mask_left] = y[0] + slope_left * (x_new[mask_left] - x[0])
        
    # Extrapolate right
    if x_new[-1] > x[-1]:
        # Calculate slope between last two points
        slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
        # Apply slope to points to the right of x[-1]
        mask_right = x_new > x[-1]
        y_new[mask_right] = y[-1] + slope_right * (x_new[mask_right] - x[-1])
        
    return y_new

def main():
    input_file = '225_W_spekpy.csv'
    output_file = '225_W_spekpy_output.csv'
    
    print(f"Reading {input_file}...")
    try:
        # Attempt to read with tab separator, skipping comment lines
        df = pd.read_csv(input_file, sep='\t', comment='#', header=None)
        # If only one column found, try comma separator
        if df.shape[1] < 2:
            df = pd.read_csv(input_file, sep=',', comment='#', header=None)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Extract energy and fluence
    # Assuming 1st column is Energy, 2nd is Fluence
    energy = df.iloc[:, 0].values
    fluence = df.iloc[:, 1].values
    
    # Ensure sorted by energy
    sort_idx = np.argsort(energy)
    energy = energy[sort_idx]
    fluence = fluence[sort_idx]
    
    # Define new integer energy buckets
    # Range from floor(min) to ceil(max) to cover the full extent
    min_e = int(np.floor(energy.min()))
    max_e = int(np.ceil(energy.max()))
    
    # Create integer array
    new_energy = np.arange(min_e, max_e + 1, dtype=int)
    
    # Perform interpolation with extrapolation
    new_intensity = extrapolate_interp(new_energy, energy, fluence)
    
    # Clip negative values to 0 (physical constraint)
    new_intensity = np.maximum(new_intensity, 0)
    
    # Normalize relative intensity (Max value = 1.0)
    max_val = new_intensity.max()
    if max_val > 0:
        new_intensity = new_intensity / max_val
        
    # Create output DataFrame
    output_df = pd.DataFrame({
        'Energy_keV': new_energy,
        'Relative_Intensity': new_intensity
    })
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    main()