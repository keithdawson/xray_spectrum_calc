# Physics review and implemented corrections

Reviewed against the original checkout at `77072bf`. The changes improve the
spectral calculation and its numerical consistency. They do **not** establish
experimental accuracy for detector fog, spatial resolution or absolute CT CNR.

## What was corrected

| Area | Finding | Implemented correction |
|---|---|---|
| Attenuation | Runtime linear interpolation introduced curvature errors despite log-interpolated source data. CT calculations used a different attenuation source. | Shared log-log interpolation, NIST coefficients and densities for filters, targets and detector. Reject unsupported energies. |
| Spectrum integration | Trapezoidal integration half-counted endpoint bins; some scripts used SpekPy's duplicated plotting coordinates. | Bin-centre spectra with consistent midpoint widths. All photons in the first/last bin are included. |
| K fluorescence | Emission used normal-path escape at every angle, rounded lines to one bin, and moved out-of-grid lines to the grid boundary. | Angle-integrated oblique escape; two-bin deposition conserving photon count and energy; omitted out-of-grid lines. |
| Atomic data | Generator caught every library error as “below edge,” yielding false zero K absorption from 300.1–500 keV. Missing K-beta caused Na/Mg to be omitted entirely. | Regenerated Z=11–92 tables over supported 1–300 keV; propagate unexpected errors; retain alpha-only emitters; preserve distinct edge anchors. |
| Compton & Rayleigh scatter | Isotropic, unchanged-energy secondary spectrum; coherent (Rayleigh) scatter removed from primary without entering accepted secondary spectrum. | Normalized Waller–Hartree (`Klein–Nishina × S(q, Z)`) incoherent angles with small-angle binding suppression, Compton recoil energy loss, oblique escape, and conservative redistribution, plus coherent (Rayleigh) elastic scatter weighted by Hubbell relativistic atomic form factors `(1 + cos²θ) |F(q, Z)|²` and `CS_Rayl` (`MT502`). |
| Detector/CNR | Optimizer treated energy signal as Poisson counts and omitted the energy-deposition weighting used by contrast analysis. | Shared detector mean and second-moment response; independent A/B noise; fixed-capacity normalization; optional inclusion of filter secondary fog in detector load and variance. |
| Optimizers | Omitted downstream fluorescence cascades and no-filter/end-point choices; geometry was hardcoded. The “emission line” objective actually retained the entire high-energy band. | Multi-layer upstream cascade generation, optional scatter-fog inclusion in fluorescence and CNR optimizers, physical filter-to-detector acceptance cone calculator, pre-filtration stack mode, configurable high-energy cutoff threshold, and precomputed log-attenuation / normalizer caches. |
| Source units | SpekPy's fluence per area was displayed as total power; selected distance only changed air absorption. | Tube-current control, inverse-square source scaling, W/cm² and photons/cm²/s labels, and persistent CSV normalization metadata. Relative spectra retain arbitrary units. |

The underlying NIST attenuation master was checked against all 92 elements'
original anchors: values matched, and no duplicate energies or nonpositive
coefficients were found. It did not need regeneration. NIST defines attenuation
as removal from a narrow beam and supports mass-fraction mixture additivity.
[NIST attenuation definitions](https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html)

The K-shell data are **Kissel** partial photoionization cross-sections from
`CS_Photo_Partial`; the former Scofield provenance and blanket <1% agreement
claims were incorrect. Data remain generated offline; neither xraylib nor
xraydb is imported by the app at runtime.
[xraylib API](https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions)

## Equations and interpretation

Let `T = density × thickness_mm / 10` in g/cm², with mass coefficients `mu` in
cm²/g. Primary transmission is `exp(-mu(E) T)`. For an outgoing direction
`c = cos(theta) > 0`, the fluorescence/scatter depth factor is

```text
J(E, E', c) = integral from 0 to T of
             exp[-mu(E) x - mu(E') (T-x)/c] dx.
```

The implementation evaluates this analytically using a cancellation-resistant
exponential difference. Isotropic fluorescence integrates `J dc/2` over the
accepted cone and multiplies by K photoionization, fluorescence yield and line
branching. Incoherent (Compton) scatter weights the Klein–Nishina differential
shape by the Waller–Hartree / Hubbell et al. (1975) atomic incoherent scattering
function `S(q, Z)` with `q = (E / 12.3984193 keV·Å) sqrt((1-c)/2)` (normalized
over `c in [-1, 1]` using a `s = sin(theta/2)` quadrature rule that removes the
endpoint square-root singularity) and `E' = E / [1 + E/(510.99895 keV) (1-c)]`.
Coherent (Rayleigh) scatter is elastic (`E' = E`) and weights the unpolarized
Thomson angular factor `0.5 (1 + c²)` by the squared relativistic Hartree–Fock
atomic form factor `|F(q, Z)|²` (`COHERENT_FF` in `fluorescence_k_data.py`),
scaled by the tabulated coherent mass cross-section `MT502` (`CS_Rayl` in
`pyne_edpl/ZA*.csv`). Because `|F(q, Z)|²` is strongly forward-peaked (`F(0, Z) = Z`),
a large fraction of Rayleigh scatter falls inside forward acceptance cones.
Partial rates are capped against the NIST total to prevent mixed-source edge
discrepancies creating photons. This bound is a numerical consistency measure,
not a resolution of source-data uncertainty. The Compton relation and
Waller–Hartree binding factor follow the
[Geant4 reference](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/compton.html)
and [Livermore Compton model](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/compton/livermore_compton.html).

For the detector, `p = 1-exp(-mu T)` and `q = E mu_en/mu`. With incident bin
counts `N`, the idealized mean is `S = sum(N p q)` and variance is
`V = sum(N p q²)`. These assume deterministic energy `q` per interaction. For
independent A/B rays and fixed air-path detector capacity `Q` in keV:

```text
CNR = sqrt(Q) |S_A-S_B| / sqrt[S_air (V_A+V_B)].
```

The optimizer reports the coefficient multiplying `sqrt(Q)`, not an absolute
CNR. This follows compound-Poisson energy-integrating statistics; the selected
deterministic response is an approximation to the required second moment.
[Energy-integrating CT statistics](https://pubmed.ncbi.nlm.nih.gov/17404467/)

`mu_en/mu` represents local energy transfer/loss, not the escape probability of
a particular finite screen. Secondary reabsorption and event-energy variation
cannot be recovered from these coefficients alone. Compound `mu_en` additivity
is also approximate because charged-particle radiative losses are not strictly
additive. [NIST energy-absorption definitions](https://physics.nist.gov/PhysRefData/XrayMassCoef/chap3.html)

## Reproducible comparisons

Run `python physics_benchmark.py`: SpekPy W, 100 kVp, 12° anode, 0.1-keV bins,
1-mm Al source filtration, 1 mAs at 100 cm, 30° full secondary acceptance.
Numbers below are integrated accepted photon fluence per cm² per mAs. The
changes reflect the revised approximation; they are not measured correction
factors for a real detector.

| Foil | Fluorescence before | Fluorescence after | Scatter (isotropic Compton) | Scatter (Compton `S(q,Z)`) | Scatter (Compton `S(q,Z)` + Rayleigh `|F(q,Z)|²`) |
|---|---:|---:|---:|---:|---:|
| Cu 0.05 mm | 196,235 | 193,992 | 24,868 | 16,116 | 709,983 |
| Mo 0.05 mm | 974,253 | 968,437 | 18,476 | 12,401 | 761,682 |
| Ta 0.05 mm | 219,385 | 218,755 | 19,478 | 11,009 | 1,798,742 |
| W 0.10 mm | 226,198 | 224,656 | 22,044 | 12,968 | 1,906,173 |

Primary transmission changes by less than 0.04% in these examples. Fluorescence
falls by approximately 0.3–1.1%. While Waller–Hartree binding (`S(0, Z) = 0`)
suppresses forward 30°-cone incoherent Compton scatter to 11,009–16,116
photons/cm²/mAs, coherent (Rayleigh) scatter has `F(0, Z) = Z` (`|F(0, Z)|² = Z²`),
which strongly concentrates elastic scatter into forward angles (`θ <= 15°`) and
makes forward-cone Rayleigh scatter a dominant component of the accepted
scattered field, particularly for high-Z filters (`Ta`, `W`).

For the detector/CNR comparison, use the same 100-kVp/12° source with 0.5-keV
bins, FR4 1 mm versus Cu 0.035 mm, and Gadox 0.2 mm. Weber contrast changes from
12.0058% to 11.8902%. Add common FR4 1.6 mm plus Sn63Pb37 solder 0.5 mm, path
multiplier 3, and maximum exposure penalty 5 for the filter search:

| Material | Previous optimum, mm | Corrected optimum, mm |
|---|---:|---:|
| Mo | 0.27189 | 0.19750 |
| Cu | 0.82061 | 0.59345 |
| Sn | 0.26699 | 0.21468 |
| Al | 15.52964 | 12.36749 |

The ranking is unchanged in this example, but optimal thickness falls 20–28%.
Old/new CNR scores have different noise definitions and cannot be compared
directly. These API benchmarks omit downstream air unless explicitly supplied.

The standalone 220-kVp, 2-mm-maximum high-energy optimizer still ranks
Zr 0.19 mm first. Its score changes by -0.082%; Tm's score changes by +3.62%
because bin-centre handling matters near its edge and the W characteristic
lines. The Sm optimum changes from 0.09 to 0.08 mm.

## Validation and recommended next work

The automated suite covers analytic slab integrals, photon/energy conservation,
K thresholds and tabulated support, KN/Thomson limits, Waller–Hartree `S(q, Z)`
forward-scatter suppression and normalization, Hubbell `F(q, Z)` coherent
Rayleigh elastic scatter and normalization, independent angular/depth
integration, scattered-line shapes, physical acceptance-cone geometry, CNR
second moments/scaling (including secondary-fog penalties), optimizer
boundaries/cascades (including multi-layer upstream stacks and scatter-aware
secondary ranking), source distance/current scaling and cache invalidation.
Both repository-required test scripts pass. Streamlit smoke runs calculate
the saved filter stack, run both CNR and fluorescence optimizers, and load a
relative input spectrum without errors. The app applies the selected downstream
air path consistently to those two optimizer objectives. Packaging scripts
include the new modules.

For quantitative industrial-CT prediction, the next improvements should be:

1. **Full 2D/3D spatial transport and collimation.** The plane-slab cone
   calculator now supports deriving acceptance angle from filter-to-detector
   distance and aperture diameter, and includes both incoherent (`S(q, Z)`) and
   coherent (`F(q, Z)`) scatter, but still treats accepted secondaries as a 1D
   spectral sum rather than a 2D spatial point-spread / glare distribution across
   the detector plane.
2. **Finite-detector response and measured noise.** Use screen packing density,
   optical response, event-level energy-deposition moments, fluorescence
   reabsorption/escape and measured gain/noise. The present variance can be too
   small at a fixed mean, and CNR rankings can change with the detector.
3. **Transport benchmark against measurement or Monte Carlo.** Include L/M
   fluorescence, repeated interactions within each layer, object/air scatter
   and angular transport between layers. Heavy-element L emission can matter
   at lower tube voltages even when K fluorescence is absent. Test foil order,
   thickness and cone width against Geant4/PENELOPE or calibrated measurements.
4. **Cross-section consistency and uncertainty.** Compare low-Z high-energy
   Kissel partial rates with independent data; replace or qualify questionable
   tails. Differences between databases are particularly sensitive near edges.
   [NIST database comparison](https://www.nist.gov/pml/note-nist-x-ray-attenuation-databases)

These follow-ups need 2D spatial transport, detector characterization or an
independent transport reference. The application now describes its supported
approximation explicitly rather than treating spectra alone as a prediction of
spatial blur or reconstructed image quality.
