import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

def plot_spectrum(csv_files):
    plt.figure(figsize=(12, 7))
    
    # Use a colormap that supports many distinct colors if we have many files
    if len(csv_files) > 10:
        # tab20 has 20 distinct colors, good for up to 20 plots
        try:
            cmap = plt.get_cmap('tab20')
            if hasattr(cmap, 'colors'):
                plt.gca().set_prop_cycle('color', cmap.colors)
        except Exception:
            pass # Fallback to default cycle
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} not found. Skipping.")
            continue
            
        try:
            # Load CSV
            df = pd.read_csv(csv_file)

            # Expecting two columns: first = x, second = y
            x_col = df.columns[0]
            y_col = df.columns[1]

            x = df[x_col].values
            y = df[y_col].values

            # Calculate stats (assuming y is spectral density e.g., photons/keV)
            dE = np.gradient(x)
            total_fluence = np.sum(y * dE)
            
            # Power calculation:
            # Integrate Energy * Intensity
            # Units: keV * (Photons/s) = keV/s
            total_power_kev_s = np.sum(x * y * dE)
            
            # Convert to Watts (Joules/s)
            # 1 keV = 1.60218e-16 Joules
            total_power_watts = total_power_kev_s * 1.60218e-16

            # Plot
            plt.plot(x, y, label=f"{os.path.basename(csv_file)}\n(Fluence: {total_fluence:.5e}, Pwr: {total_power_watts:.5e} W)", alpha=0.7)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity")
    plt.title("X-ray Spectrum Comparison")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Adjust legend placement if there are many files
    if len(csv_files) > 5:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize='small')
        plt.tight_layout()
    else:
        plt.legend()
        
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot X-ray spectra from CSV files.")
    parser.add_argument("files", nargs="*", help="CSV files to plot (default: filtered_spectrum.csv)")
    
    args = parser.parse_args()
    
    files_to_plot = args.files
    if not files_to_plot:
        files_to_plot = ['filtered_spectrum.csv']
        print(f"No file specified. Plotting default: {files_to_plot[0]}")
        
    plot_spectrum(files_to_plot)
