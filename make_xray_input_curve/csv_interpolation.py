import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import argparse
import sys

def interpolate_missing_values(input_file, output_file=None, method='cubic'):
    """
    Read a CSV file and interpolate missing values in the second column.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional)
    method (str): Interpolation method - 'linear', 'cubic', 'quadratic', or 'spline'
    """
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if there are at least 2 columns
        if df.shape[1] < 2:
            raise ValueError("CSV must have at least 2 columns")
        
        # Get the second column (index 1)
        column_name = df.columns[1]
        print(f"Processing column: '{column_name}'")
        
        # Count missing values
        missing_count = df[column_name].isna().sum()
        print(f"Missing values found: {missing_count}")
        
        if missing_count == 0:
            print("No missing values to interpolate!")
            return df
        
        # Create a copy of the dataframe
        df_interpolated = df.copy()
        
        # Get non-null values and their indices
        non_null_mask = ~df[column_name].isna()
        non_null_indices = df.index[non_null_mask].values
        non_null_values = df[column_name][non_null_mask].values
        
        # Check if we have enough non-null values for interpolation
        if len(non_null_values) < 2:
            print("Warning: Need at least 2 non-null values for interpolation")
            return df
        
        # Create interpolation function
        if method == 'cubic' and len(non_null_values) >= 4:
            # Cubic spline interpolation
            interp_func = interp1d(non_null_indices, non_null_values, 
                                 kind='cubic', bounds_error=False, 
                                 fill_value='extrapolate')
        elif method == 'quadratic' and len(non_null_values) >= 3:
            # Quadratic interpolation
            interp_func = interp1d(non_null_indices, non_null_values, 
                                 kind='quadratic', bounds_error=False, 
                                 fill_value='extrapolate')
        else:
            # Fall back to linear interpolation
            interp_func = interp1d(non_null_indices, non_null_values, 
                                 kind='linear', bounds_error=False, 
                                 fill_value='extrapolate')
            if method != 'linear':
                print(f"Warning: Not enough points for {method} interpolation, using linear instead")
        
        # Interpolate missing values
        all_indices = df.index.values
        interpolated_values = interp_func(all_indices)
        
        # Fill in the missing values
        df_interpolated[column_name] = interpolated_values
        
        print(f"Successfully interpolated {missing_count} missing values")
        
        # Save to output file if specified
        if output_file:
            df_interpolated.to_csv(output_file, index=False)
            print(f"Saved interpolated data to: {output_file}")
        else:
            # Generate default output filename
            base_name = input_file.rsplit('.', 1)[0]
            default_output = f"{base_name}_interpolated.csv"
            df_interpolated.to_csv(default_output, index=False)
            print(f"Saved interpolated data to: {default_output}")
        
        return df_interpolated
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Interpolate missing values in CSV second column')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('-m', '--method', choices=['linear', 'cubic', 'quadratic'], 
                       default='cubic', help='Interpolation method (default: cubic)')
    parser.add_argument('--preview', action='store_true', 
                       help='Show before/after preview of first 10 rows')
    
    args = parser.parse_args()
    
    # Store original data for comparison if preview requested
    if args.preview:
        try:
            original_df = pd.read_csv(args.input_file)
        except:
            original_df = None
    
    # Perform interpolation
    result_df = interpolate_missing_values(args.input_file, args.output, args.method)
    
    # Show preview if requested
    if args.preview and result_df is not None and original_df is not None:
        print("\n" + "="*50)
        print("BEFORE INTERPOLATION (first 10 rows):")
        print("="*50)
        print(original_df.head(10))
        
        print("\n" + "="*50)
        print("AFTER INTERPOLATION (first 10 rows):")
        print("="*50)
        print(result_df.head(10))

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("CSV Interpolation Script")
        print("Usage examples:")
        print("  python script.py data.csv")
        print("  python script.py data.csv -o output.csv")
        print("  python script.py data.csv -m linear --preview")
        print("  python script.py data.csv -m cubic -o smoothed_data.csv")
    else:
        main()