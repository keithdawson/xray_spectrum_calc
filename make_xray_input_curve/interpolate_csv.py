import pandas as pd
import numpy as np
import argparse
import sys

def interpolate_missing_data(input_file, output_file, col_index=1):
    """
    Reads a CSV file, performs linear interpolation on a specified column
    to fill missing values, and saves the result to a new CSV file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to save the output CSV file.
        col_index (int): The index of the column to interpolate (0-based).
    """
    try:
        # Read the CSV file into a pandas DataFrame.
        # We assume no header, but you could adjust this with the `header` parameter.
        df = pd.read_csv(input_file, header=None)
        print(f"Successfully loaded '{input_file}'.")
        print("Original Data:")
        print(df)
        print("-" * 20)

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        sys.exit(1)

    if col_index >= len(df.columns):
        print(f"Error: Column index {col_index} is out of bounds for the given CSV file.")
        sys.exit(1)

    # Use the first column as our x-values (e.g., time or sequence)
    # and the specified column as our y-values.
    x = df.iloc[:, 0]
    y = df.iloc[:, col_index]

    # Convert the column to numeric, coercing errors to NaN (Not a Number)
    y_numeric = pd.to_numeric(y, errors='coerce')

    # Find the indices of the data points that are not NaN
    not_nan_indices = np.where(~np.isnan(y_numeric))[0]

    if len(not_nan_indices) < 2:
        print("Error: Need at least two data points to perform interpolation.")
        sys.exit(1)

    # Get the x and y values of the known data points
    xp = x[not_nan_indices]
    fp = y_numeric[not_nan_indices]

    # Find the indices of all data points (including missing ones)
    all_indices = np.arange(len(x))

    # Perform linear interpolation
    # np.interp finds the interpolated values at `all_indices`
    # using the known points (xp, fp).
    interpolated_values = np.interp(all_indices, xp, fp)

    # Create a copy of the DataFrame to avoid modifying the original one in place
    df_interpolated = df.copy()

    # Replace the original column with the new, interpolated data
    df_interpolated[col_index] = interpolated_values

    print("Interpolated Data:")
    print(df_interpolated)
    print("-" * 20)

    try:
        # Save the new DataFrame to the output file, without the index.
        df_interpolated.to_csv(output_file, index=False, header=False)
        print(f"Successfully saved interpolated data to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="Fill missing values in a CSV column using linear interpolation."
    )
    parser.add_argument("input_file", help="The path to the input CSV file.")
    parser.add_argument("output_file", help="The path to save the output CSV file.")
    parser.add_argument(
        "--col",
        type=int,
        default=1,
        help="The 0-based index of the column to interpolate (default: 1).",
    )

    args = parser.parse_args()

    # Run the main function with the provided arguments
    interpolate_missing_data(args.input_file, args.output_file, args.col)
