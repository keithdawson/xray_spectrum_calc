# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

X-ray filter-stack modeling tool for industrial CT of electronics (PCBs).
Calculates spectrum attenuation through filter stacks plus the secondary
radiation the filters emit (K fluorescence and Compton scatter fog), and
optimizes filter choices for contrast/CNR. See readme.md for the physics
rationale.

## Running

- App: `streamlit run app.py` (or via `.claude/launch.json`, port 8510)
- Tests: `python test_fluorescence_physics.py` (unit, mock data),
  `python test_fluorescence_validation.py` (integration, needs spekpy)

## Architecture

- `apply_filter_and_fluorescence.py` — physics engine. `process_single_layer()`
  returns `(phi_total, phi_fluorescence, phi_scatter, transmission_factor)`;
  `run_stack()` chains layers and writes `_fluoro.csv` / `_scatter.csv`.
  Also home of material constants (`DENSITIES`, `SYM_TO_Z`,
  `PRACTICAL_FILTER_MATERIALS`).
- `app.py` — Streamlit UI. Persists filter stack to `.last_filters.json`
  and the enabled-materials selection to `.material_settings.json`.
- `ct_optimizer.py` — fluorescence-suppression and CNR filter optimizers.
- `ct_physics.py` — target contrast + scintillator energy deposition
  (weights deposited energy by mu_en/mu).
- `emission_optimizer.py` — sweeps filters to isolate W K-lines (>= 57 keV);
  caches results to `emission_opt_results_*.csv`.

## Data (all generated — do not hand-edit)

- `attenuation_coefficients_long.csv` — NIST mu/rho and mu_en/rho, all 92
  elements, log-log interpolated with exact absorption-edge steps.
  Regenerate: `python extract_atten_coeff/regenerate_master_from_nist.py`
  (reads the local mirror in `physics.nist.gov/`).
- `pyne_edpl/ZA*.csv` — K-shell photoionization (tauK) and incoherent
  scatter (MT504) per element, 0.1 keV grid, Z=11-92, from xraylib.
  Regenerate: `python generate_fluorescence_xs.py` (needs `pip install xraylib`).
- `fluorescence_k_data.py` — K yields (Krause), weighted Ka/Kb line energies,
  branching ratios, atomic masses.
  Regenerate: `python generate_fluorescence_data.py` (needs xraylib).

xraylib/xraydb are generator-time dependencies only; the app does not
import them at runtime.

## Conventions

- Spectra CSVs use columns `Energy_KeV`, `Photons_per_s` (fluence density
  per keV). SpekPy-generated spectra default to 0.1 keV bins (`dk`).
- Mass attenuation is cm^2/g, thickness mm (converted to g/cm^2 internally).
- When changing physics or data, run both test scripts and compare
  fluorescence/scatter magnitudes before/after — optimizer rankings are
  sensitive to few-percent shifts.
