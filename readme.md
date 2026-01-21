
# Xray Absorption Calculator [![Xrays Pew Pew](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme)

A toolset for calculating X-ray spectrum attenuation through filter stacks, including the generation of characteristic fluorescence radiation from the filters themselves.

## Features

- **Filter Attenuation**: Calculates transmission through multiple layers of materials (Mo, Al, Cu, Sn, etc.).
- **Fluorescence Generation**: Calculates K-alpha and K-beta fluorescence emitted by filters when excited by the beam.
- **EPDL Integration**: Uses extracted EPDL (Evaluated Photon Data Library) data for accurate photoelectric cross-sections.
- **Geometry Awareness**: Models fluorescence acceptance based on a detector geometry (solid angle).

## Scripts & Usage

### 1. Main Calculator (`apply_filter_and_flourescence.py`)
The core script. Takes an input spectrum, applies a stack of filters, calculates attenuation and fluorescence, and outputs the resulting spectrum.

**Usage:**
```bash
python apply_filter_and_flourescence.py \
  --spectrum 225_W_spekpy_output.csv \
  --filter-materials Mo Al Air \
  --filter-thicknesses-mm 0.25 0.5 1000 \
  --out filtered_spectrum.csv
```

### 2. Plotting (`plot_csv.py`)
Visualizes one or more spectrum CSV files. Calculates total fluence and photon power.

**Usage:**
```bash
# Plot single file
python plot_csv.py filtered_spectrum.csv

# Compare input vs output
python plot_csv.py 225_W_spekpy_output.csv filtered_spectrum.csv
```

### 3. SpekPy Converter (`convert_spekpy.py`)
Converts SpekPy output (often irregular bins) into the standard CSV format used by this tool (integer keV bins, normalized intensity).

**Usage:**
```bash
# Reads 225_W_spekpy.csv and writes 225_W_spekpy_output.csv
python convert_spekpy.py
```

### 4. Data Generation (`generate_master_attenuation.py`)
Compiles individual element attenuation CSVs (extracted from NIST HTML) into a single master lookup file.

**Usage:**
```bash
python generate_master_attenuation.py
```

### 5. HTML Scraper (`convert_html_tables_to_csv.py`)
Parses NIST HTML files in the `extract_atten_coeff` folder to create individual CSVs for `generate_master_attenuation.py`. Handles absorption edge steps by creating infinitesimal energy increments.

**Usage:**
```bash
python convert_html_tables_to_csv.py
```

## Testing

We have created specific tests to verify the physics calculations, particularly for fluorescence which is complex to model.

### `test_fluorescence_physics.py`
**Purpose:** Unit test for the physics logic.
**What it does:**
1.  Creates a mock material (Mo) and mock spectrum.
2.  **Test 1 (Below Edge):** Sends 15 keV photons (below Mo K-edge of ~20 keV). Verifies **zero** fluorescence is produced.
3.  **Test 2 (Above Edge):** Sends 30 keV photons. Verifies fluorescence peaks appear at correct K-alpha and K-beta energies.
4.  **Physics Check:** Manually calculates expected photon counts based on cross-sections, yield, and geometry to ensure the script's math matches the physics formula.

**Run:**
```bash
python test_fluorescence_physics.py
```

### `test_fluorescence_beam.py`
**Purpose:** Integration test with visual verification.
**What it does:**
1.  Runs the filter logic but forces the geometric acceptance to 100% (as if all fluorescence went forward).
2.  This exaggerates the fluorescence peaks so they are clearly visible on a plot.
3.  Automatically plots the result against the baseline to visually confirm peak locations (e.g., Mo peaks at 17.4 and 19.6 keV).

**Run:**
```bash
python test_fluorescence_beam.py
```

## Data Sources

- **Attenuation**: NIST X-ray Mass Attenuation Coefficients (scraped from local HTML files).
- **Fluorescence Data**: EPDL (Evaluated Photon Data Library) extracted CSVs located in `pyne_edpl/`.

## Folder Structure

- `pyne_edpl/`: Contains extracted EPDL data (cross-sections).
- `extract_atten_coeff/`: Contains NIST HTML files and generation scripts.
