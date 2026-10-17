
# Xray Absorption Calculator [![Xrays Pew Pew](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme)

A toolset for calculating X-ray spectrum attenuation through filter stacks, including the generation of characteristic fluorescence radiation from the filters themselves.

## Features

- **Filter Attenuation**: Calculates transmission through multiple layers of materials (all elements Z=1-92, NIST data with exact absorption-edge steps).
- **Fluorescence Generation**: Calculates K-alpha and K-beta fluorescence emitted by filters when excited by the beam, for any filter element Z=11-92 (Kissel K-shell cross-sections, Krause yields, weighted line energies; 1–300 keV).
- **Compton Scatter**: Single-collision transport with Klein–Nishina angular weighting, Compton energy loss and oblique escape. Uses binding-corrected total cross-sections; the angular shape remains a free-electron approximation.
- **Acceptance Geometry**: Integrates secondary radiation over a specified forward cone through a plane slab. This is a directional acceptance parameter, not a detector geometry or focal-spot blur calculation.
- **CT Context**: Shared NIST target attenuation and an idealized first-interaction scintillator response weighted by mu_en/mu. CNR uses energy-weighted compound-Poisson variance. Finite-screen escape/reabsorption and detector noise require further validation.

## Project Goals & CT Physics

This tool is specifically tailored toward optimizing X-ray beam profiles for **Computed Tomography (CT) scanning** of high-density electronics (e.g., heavily populated PCBs containing FR4, Copper, Tin, and Lead). The physical modeling revolves around two major optimizations:

1. **Beam Hardening for Contrast**: Raw tungsten emissions contain too many soft X-rays, which lack the energy to penetrate dense solder components (Tin/Lead) and result in detector saturation without adding useful contrast. This app uses complex filter stacks to mathematically "harden" the beam, ensuring enough high-energy photons penetrate the board to maximize the contrast of the critical elements (e.g., differentiating Copper traces from FR4 substrate) without widening the detector histogram unnecessarily.
2. **Reducing Filter-Generated Secondaries**: Filters emit K fluorescence and scatter photons, which can degrade imaging. This code estimates their accepted energy spectra in a plane-parallel slab model. It does not predict their spatial distribution, focal-spot enlargement, modulation transfer, or reconstructed CT image quality.

See [PHYSICS_REVIEW.md](PHYSICS_REVIEW.md) for implemented corrections, equations, before/after benchmarks, sources and prioritized next improvements. The numerical checks establish consistency with the stated model; they are not an experimental validation of detector fog or absolute CNR.

### Units and supported range

Spectra contain densities at strictly increasing positive energy-bin centres. Integrals use midpoint-bin widths, including full endpoint bins. SpekPy `edges=True` produces a plotting staircase and must not be used for calculations.

Generated app spectra use tube current in mA and source distance in mm, returning photons/cm²/s/keV. Power is W/cm². Air attenuation is applied separately. Imported spectra retain their normalization: `Photons_per_s` defaults to photons/s/keV; `Relative_Intensity` is arbitrary units. Exported `Normalization` metadata records this distinction.

The attenuation table covers 1–500 keV. The regenerated Z=11–92 K-shell and incoherent tables cover 1–300 keV; the full secondary model is supported in that range. Requests outside table support raise errors instead of silently extrapolating. Photoionization data are Kissel partial cross-sections, not Scofield. Low-Z high-energy tails and mixed-source differences near edges still require independent benchmarking.

The cone accepts downstream directions only (full angle 0–180 degrees). Secondary rays propagate axially between layers; intra-layer re-interactions, L/M fluorescence, Rayleigh additions, object scatter and air-generated secondaries are omitted. Lines shifted outside the represented centre range leave the calculation.

## Interactive App (`app.py`)

The primary interface — a Streamlit app wrapping the physics engine:

```bash
pip install -r requirements.txt
streamlit run app.py
```

- **Spectrum input**: generate with SpekPy (kVp, anode angle/material, 0.1 keV default bins) or load/upload a CSV.
- **Filter stack editor**: reorderable material + thickness rows, persisted between sessions (`.last_filters.json`).
- **Results**: power/flux at the DUT with run-over-run deltas, fluorescence and Compton scatter power shown separately (increases flagged red — more fog is bad), log-scale and display-rebinning chart controls, CSV downloads for all three components.
- **Material selection** (⚙️ Materials): choose which foil elements the dropdowns and optimizers use, from a curated list of 36 practically-purchasable materials; persisted in `.material_settings.json`.
- **Optimizers**: Cascading Fluorescence (secondary filter to absorb the primary's fluorescence), CNR Filter (penetration vs contrast under shadowing), High-Energy Fluence (preserve the entire band >= 57 keV, including continuum). Each has one-click "Apply" buttons to push results into the filter stack.
- **Contrast analysis**: energy deposited in the scintillator through Target A vs Target B, with the |A-B| difference trace showing where the contrast comes from.

Offline installers for machines without internet can be built with `package_windows.ps1` / `package_linux.ps1` (bundles Python + wheels into `deploy_windows/` / `deploy_linux/`).

## Scripts & Usage

### 1. Main Calculator (`apply_filter_and_fluorescence.py`)
The core script. Takes an input spectrum, applies a stack of filters, calculates attenuation, fluorescence, and Compton scatter, and outputs the resulting spectrum. With `--save-fluorescence` it also writes the isolated secondary components (`*_fluoro.csv`, `*_scatter.csv`).

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
Converts SpekPy output (often irregular bins) into the standard CSV format used by this tool (integer keV bins, normalized relative intensity; not an absolute photon-rate calibration).

This legacy converter interpolates and normalizes for qualitative plots. For quantitative calculations, use the app's direct SpekPy generation or export native bin centres and fluence densities; resampling characteristic peaks with this converter does not conserve their integrated strength.

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
Regenerates `fluorescence_k_data.py`: K-shell fluorescence yields (Krause 1979), rad-rate-weighted K-alpha/K-beta line energies, branching ratios, and atomic masses for Z=11-92, extracted from `xraylib`. Only needed to refresh the data module; the app does not require xraylib at runtime.

**Usage:**
```bash
pip install xraylib
python generate_fluorescence_data.py
```

### 6. Cross-Section Generation (`generate_fluorescence_xs.py`)
Regenerates the per-element files in `pyne_edpl/` (K-shell photoionization `tauK` and incoherent scatter `MT504`) from xraylib on a 0.1 keV grid from 1–300 keV with sharp K-edge onsets, for Z=11-92. Out-of-range library errors are propagated rather than replaced by zeros.

**Usage:**
```bash
pip install xraylib
python generate_fluorescence_xs.py
```

## Testing

```bash
python test_fluorescence_physics.py
python test_fluorescence_validation.py
python -m unittest discover -v
python physics_benchmark.py
```

The tests cover analytic Beer–Lambert and slab integrals, K-edge thresholds, line photon/energy conservation, Compton angular normalization and energy redistribution, real-data bounds, CNR second moments and scaling, optimizer boundaries/cascades, cache invalidation, and source inverse-square/current scaling. `physics_benchmark.py` reproduces the foil comparisons in the review. SpekPy is required for integration and source tests; xraylib remains a generator-time dependency only.

`test_fluorescence_beam.py` is a visual diagnostic using the production engine with a 180-degree forward acceptance cone.

## Data Sources

- **Attenuation**: NIST X-ray Mass Attenuation Coefficients (regenerated from the local HTML mirror with log-log interpolation and exact edge steps; see `extract_atten_coeff/regenerate_master_from_nist.py`).
- **Cross-sections** (`pyne_edpl/ZA*.csv`): Kissel K-shell photoionization (`CS_Photo_Partial`) and incoherent scatter (`CS_Compt`), generated from xraylib on a 0.1 keV grid with sharp K-edge onsets for Z=11–92, 1–300 keV (`generate_fluorescence_xs.py`). These replace older EPDL extracts; no blanket sub-percent agreement is claimed.
- **Fluorescence lines** (`fluorescence_k_data.py`): Krause (1979) K yields, rad-rate-weighted line energies and branching ratios from xraylib (`generate_fluorescence_data.py`).

## Folder Structure

- `pyne_edpl/`: Per-element cross-section CSVs (see Data Sources).
- `extract_atten_coeff/`: NIST HTML mirror extraction and master-table generation scripts.
