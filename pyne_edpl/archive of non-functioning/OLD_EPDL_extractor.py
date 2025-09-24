import numpy as np
import csv
import os
from pyne import endl

def extract_data_with_pyne(input_txt_file, output_csv_file, nuc_id):
    """
    Extracts cross-section data from an EPDL txt file using the pyne library
    and saves it as a CSV.

    It stops reading data when energy exceeds 450 keV.

    Args:
        input_txt_file (str): Path to the input EPDL .txt file.
        output_csv_file (str): Path for the output .csv file.
        nuc_id (int): The nucleus ID for the data library (e.g., 42000 for Mo).
    """
    print(f"Processing {input_txt_file} with pyne...")

    if not os.path.exists(input_txt_file):
        print(f"Error: Input file not found at '{input_txt_file}'")
        return

    try:
        # Open the EPDL file using pyne's endl Library class
        lib = endl.Library(input_txt_file)

        # Extract the different types of cross-section data
        # yo=0 -> photon, yi=1 -> coherent, yi=2 -> incoherent, yi=3 -> photoelectric, yi=4 -> pair production
        coherent_data = lib.get_rx(nuc_id, yo=0, yi=1)
        incoherent_data = lib.get_rx(nuc_id, yo=0, yi=2)
        photoelectric_data = lib.get_rx(nuc_id, yo=0, yi=3)
        pair_production_data = lib.get_rx(nuc_id, yo=0, yi=4)
        
        # The first column is energy (in MeV), the second is the cross-section
        energies_mev = coherent_data[:, 0]
        coherent_xs = coherent_data[:, 1]
        incoherent_xs = incoherent_data[:, 1]
        photoelectric_xs = photoelectric_data[:, 1]
        pair_production_xs = pair_production_data[:, 1]

        # Filter the data to only include energies up to 450 keV (0.45 MeV)
        mask = energies_mev <= 0.45
        energies_kev = (energies_mev[mask] * 1000.0)

        if len(energies_kev) == 0:
            print(f"Warning: No data below 450 keV found in {input_txt_file}.")
            return

        # Write the extracted data to the output CSV file.
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['energy_keV', 'coherent_barn_per_atom',
                             'incoherent_barn_per_atom', 'photoelectric_barn_per_atom',
                             'pair_production_barn_per_atom'])
            for i in range(len(energies_kev)):
                writer.writerow([energies_kev[i],
                                 coherent_xs[mask][i],
                                 incoherent_xs[mask][i],
                                 photoelectric_xs[mask][i],
                                 pair_production_xs[mask][i]])
        print(f"Successfully wrote data to {output_csv_file}")

    except Exception as e:
        print(f"An error occurred while processing {input_txt_file}: {e}")
        print("Please ensure the 'pyne' library is installed ('pip install pyne').")


if __name__ == '__main__':
    # Process the data for Molybdenum (Mo, Z=42)
    # The nuc_id for Mo is 42000 (Z=42, A=000 for natural abundance)
    mo_input_file = 'ZA042000.txt'
    mo_output_file = 'mo_mu_tau_from_epdl.csv'
    extract_data_with_pyne(mo_input_file, mo_output_file, 42000)

    # Process the data for Lead (Pb, Z=82)
    # The nuc_id for Pb is 82000 (Z=82, A=000 for natural abundance)
    pb_input_file = 'ZA082000.txt'
    pb_output_file = 'pb_mu_tau_from_epdl.csv'
    extract_data_with_pyne(pb_input_file, pb_output_file, 82000)

    print("\nProcessing complete.")

