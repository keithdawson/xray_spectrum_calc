
# Xray Absorption Calculator [![Xrays Pew Pew](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme)

A toolset for calculating X-ray spectrum attenuation through filter stacks, including the generation of characteristic fluorescence radiation from the filters themselves.

## Features

- **Filter Attenuation**: Calculates transmission through multiple layers of materials (Mo, Al, Cu, Sn, etc.).
- **Fluorescence Generation**: Calculates K-alpha and K-beta fluorescence emitted by filters when excited by the beam.
- **Compton Scatter Fog**: Models incoherent scatter from the filters as a secondary extended source (same cone-acceptance geometry as fluorescence), using EPDL incoherent cross-sections with a Klein-Nishina fallback.
- **EPDL Integration**: Uses extracted EPDL (Evaluated Photon Data Library) data for accurate photoelectric cross-sections.
- **Geometry Awareness**: Models fluorescence acceptance based on the source beam cone angle.

## Project Goals & CT Physics

This tool is specifically tailored toward optimizing X-ray beam profiles for **Computed Tomography (CT) scanning** of high-density electronics (e.g., heavily populated PCBs containing FR4, Copper, Tin, and Lead). The physical modeling revolves around two major optimizations:

1. **Beam Hardening for Contrast**: Raw tungsten emissions contain too many soft X-rays, which lack the energy to penetrate dense solder components (Tin/Lead) and result in detector saturation without adding useful contrast. This app uses complex filter stacks to mathematically "harden" the beam, ensuring enough high-energy photons penetrate the board to maximize the contrast of the critical elements (e.g., differentiating Copper traces from FR4 substrate) without widening the detector histogram unnecessarily.
2. **Minimizing Secondary Fluorescence (Focal Spot Sharpness)**: In CT, image sharpness is governed by the size of the X-ray focal spot. When a filter is placed in the beam path, it absorbs X-rays and re-emits isotropic **fluorescence**. This effectively turns the entire illuminated area of the filter into a massive, blurry secondary X-ray source. The application explicitly models this scattering physics: the primary focal-spot beam acts as static *Fluence* against a central pixel, while the secondary fluorescence dynamically scales with the physical collimator (cone) angle. By isolating these components, the user can visually and mathematically optimize the filter stack to minimize this destructive secondary glow.

## Scripts & Usage

### 1. Main Calculator (`apply_filter_and_fluorescence.py`)
The core script. Takes an input spectrum, applies a stack of filters, calculates attenuation and fluorescence, and outputs the resulting spectrum.

**Usage:**
```bash
python apply_filter_and_fluorescence.py \
  --spectrum 225_W_spekpy_output.csv \
  --filter-materials Mo Al Air \
  --filter-thicknesses-mm 0.25 0.5 1000 \
  --out filtered_spectrum.csv \
  --save-fluorescence
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

### 4. Attenuation Data Generation (`extract_atten_coeff/regenerate_master_from_nist.py`)
Regenerates `attenuation_coefficients_long.csv` (all 92 elements) directly from the local NIST XrayMassCoef HTML mirror in `physics.nist.gov/`. Uses log-log interpolation between NIST anchor energies (mu/rho follows a power law, so linear interpolation overestimates by up to ~10% in the sparse 60-300 keV region) and preserves absorption edges as sharp steps at their true energies.

**Usage:**
```bash
python extract_atten_coeff/regenerate_master_from_nist.py
```

### 5. Fluorescence Data Generation (`generate_fluorescence_data.py`)
Regenerates `fluorescence_k_data.py`: K-shell fluorescence yields (Krause 1979), rad-rate-weighted K-alpha/K-beta line energies, and branching ratios for Z=11-92, extracted from `xraylib`. Only needed to refresh the data module; the app does not require xraylib at runtime.

**Usage:**
```bash
pip install xraylib
python generate_fluorescence_data.py
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

- **Attenuation**: NIST X-ray Mass Attenuation Coefficients (regenerated from the local HTML mirror with log-log interpolation and exact edge steps; see `extract_atten_coeff/regenerate_master_from_nist.py`).
- **Cross-sections** (`pyne_edpl/ZA*.csv`): K-shell photoionization (Scofield) and incoherent scatter, generated from xraylib on a 0.1 keV grid with sharp K-edge onsets for Z=11-92 (`generate_fluorescence_xs.py`). These replaced the original coarse-grid EPDL extracts; both derive from Scofield's calculations and agree to <1%.
- **Fluorescence lines** (`fluorescence_k_data.py`): Krause (1979) K yields, rad-rate-weighted line energies and branching ratios from xraylib (`generate_fluorescence_data.py`).

## Folder Structure

- `pyne_edpl/`: Per-element cross-section CSVs (see Data Sources).
- `extract_atten_coeff/`: NIST HTML mirror extraction and master-table generation scripts.
