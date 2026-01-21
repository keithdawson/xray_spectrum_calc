import os
import glob
import re
import csv

def process_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the content inside <PRE> tags
    # The NIST files usually wrap the ASCII table in <PRE>
    match = re.search(r'<PRE>(.*?)</PRE>', content, re.DOTALL | re.IGNORECASE)
        
    if not match:
        print(f"Skipping {filepath}: No <PRE> tag found.")
        return

    pre_content = match.group(1)
    lines = pre_content.strip().split('\n')

    data_rows = []
    
    # Regex to match data lines (scientific notation numbers)
    # e.g. 1.00000E-03  2.211E+03  2.209E+03 
    # Captures 3 groups of numbers
    data_line_pattern = re.compile(r'^\s*(?:[a-zA-Z0-9]+\s+)?([0-9\.]+E[+-][0-9]+)\s+([0-9\.]+E[+-][0-9]+)\s+([0-9\.]+E[+-][0-9]+)\s*$')

    last_energy = None

    for line in lines:
        # Check if line matches data pattern
        m = data_line_pattern.match(line)
        if m:
            row = list(m.groups())
            current_energy = float(row[0])
            
            if last_energy is not None and current_energy == last_energy:
                # Increment at the smallest MeV digit (approx 5th decimal place in sci notation)
                modified_energy = current_energy * 1.00001
                row[0] = "{:.5E}".format(modified_energy)
            
            last_energy = current_energy
            data_rows.append(row)

    if not data_rows:
        print(f"Skipping {filepath}: No matching data rows found.")
        return

    # Output CSV filename
    csv_filename = os.path.splitext(filepath)[0] + '.csv'
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header matching the user's manual format
        writer.writerow(['Energy_MeV', 'Mu_over_rho', 'mu_en_over_rho'])
        # Write data
        writer.writerows(data_rows)
    
    print(f"Generated {csv_filename}")

def main():
    # Target folder containing the HTML files
    # Assuming the script is in the parent directory of 'extract_atten_coeff'
    target_folder = 'extract_atten_coeff'
    
    # Check if folder exists, otherwise check current directory
    if not os.path.exists(target_folder):
        if glob.glob('*.html'):
            target_folder = '.'
        else:
            print(f"Folder '{target_folder}' not found and no HTML files in current directory.")
            return
    
    html_files = glob.glob(os.path.join(target_folder, '*.html'))
    
    print(f"Found {len(html_files)} HTML files in '{target_folder}'. Processing...")

    for html_file in html_files:
        process_html_file(html_file)

if __name__ == "__main__":
    main()
