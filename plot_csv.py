import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_spectrum(csv_file=None):
    # If no file was passed in, use default
    if csv_file is None:
        csv_file = 'filtered_spectrum.csv'
        print(f"No file specified. Plotting default: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)

    # Expecting two columns: first = x, second = y
    x_col = df.columns[0]
    y_col = df.columns[1]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], label=csv_file)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("X-ray Spectrum")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # If the script was run with an argument, use it; else use default
    if len(sys.argv) > 1:
        plot_spectrum(sys.argv[1])
    else:
        plot_spectrum()
