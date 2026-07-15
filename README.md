# GST225 laser crystallization model

Python implementation and diagnostics for the thermal/crystallization model
discussed in:

T. Kunkel et al., *Crystallization of GST225 thin film induced by a single
femtosecond laser pulse: Experimental and theoretical study*, Materials Science
in Semiconductor Processing 139 (2022) 106350.

The code is a research reproduction/diagnostic tool, not the authors' original
program.

## What is implemented

- 2D axisymmetric heat conduction in a GST film on a homogeneous substrate.
- Gaussian 1030 nm pulse with Beer-Lambert absorption.
- Optional melting/freezing latent heat through an equivalent heat-capacity
  method.
- Article-like crystallinity maps `Xc(r,z)` using digitized TTT curves.
- Diagnostics for fluence fitting, article temperature-profile digitization,
  convergence, and comparison with COMSOL point traces.

The heat source uses

```text
F(r) = 2 E / (pi w^2) exp(-2 r^2 / w^2)
q(r,z) = (1 - R) alpha F(r) exp(-alpha_eff z)
```

where `w` is the 1/e² fluence radius. Fitting the article screenshot gives
`w ≈ 35 um` and `F0 ≈ 34 mJ/cm²` for `650 nJ`.

Current default GST optical parameters:

```text
alpha_linear = 10 um^-1
alpha_eff    = 16.2 um^-1
```

## Setup

Create/activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Dependencies are:

```text
numpy
scipy
matplotlib
pillow
```

Run tests:

```powershell
python -m unittest -v
```

## Main article-like run with digitized TTT curves

The current main script is:

[run_article_case_digitized_ttt.py](run_article_case_digitized_ttt.py)

Run:

```powershell
.\.venv\Scripts\python.exe run_article_case_digitized_ttt.py
```

By default it uses WebPlotDigitizer data from:

```text
output/wpd_datasets.csv
```

Important options:

```powershell
--energy-nj 650
--film-nm 230
--substrate silica
--end-ns 450
--dt-ns 0.5
--no-phase-change
--interpolate-ttt
--min-delta-t-for-crystallinity-k 350
```

TTT interpolation is off by default. Add `--interpolate-ttt` only if you want
continuous values between digitized TTT contours.

Main outputs:

```text
output/article_case_digitized_ttt.png
output/figure_4c_digitized_ttt.png
output/cooling_traces_digitized_ttt.png
output/crystallinity_profile_digitized_ttt.png
output/crystallinity_profile_digitized_ttt.npz
output/crystallinity_profile_digitized_ttt.mat
```

The colorbar in crystallinity maps is fixed to `0..1`.

## Original formula-based model

[run_article_case.py](run_article_case.py) still runs the formula-based kinetic
model:

```powershell
.\.venv\Scripts\python.exe run_article_case.py
```

This route calculates TTT curves from nucleation/growth equations and is useful
as a diagnostic, but it does not reproduce the article TTT contours perfectly
with the printed parameters.

## Fluence fit from the article image

[fit_article_fluence_from_image.py](fit_article_fluence_from_image.py) digitizes
the red fluence curve in:

```text
Screenshot 2026-07-15 121007.png
```

Run:

```powershell
.\.venv\Scripts\python.exe fit_article_fluence_from_image.py
```

Output:

```text
output/fluence_fit_from_image/fluence_fit.png
output/fluence_fit_from_image/digitized_fluence.csv
```

The fit values are also printed on the plot.

## Article temperature-profile comparison

[digitize_article_temperature_profiles.py](digitize_article_temperature_profiles.py)
extracts vertical and lateral temperature profiles from the article screenshot.
It writes both raw RGB-derived temperatures and a corrected profile where the
large red plateau is mapped to the melting temperature.

```powershell
.\.venv\Scripts\python.exe digitize_article_temperature_profiles.py
```

Then compare article profiles with the Python thermal model:

```powershell
.\.venv\Scripts\python.exe compare_article_temperature_profile.py
```

The comparison plot shows both raw and corrected article points.

Output:

```text
output/article_temperature_profile_comparison.png
```

## COMSOL comparison

COMSOL point traces are read from:

```text
test_radial.txt
```

Run:

```powershell
.\.venv\Scripts\python.exe compare_comsol_radial.py
```

Outputs are saved in:

```text
output/comsol_comparison/
```

If the COMSOL model used a different beam radius, pass it explicitly. For
example, for `w = 30 um`:

```powershell
.\.venv\Scripts\python.exe compare_comsol_radial.py --spot-radius-um 30
```

There is also a phase-change smoothing diagnostic:

```powershell
.\.venv\Scripts\python.exe compare_melt_smoothing_width.py
```

## Convergence study

The convergence study is optional and never runs automatically.

Quick check:

```powershell
.\.venv\Scripts\python.exe convergence_study.py --quick
```

Full comparison:

```powershell
.\.venv\Scripts\python.exe convergence_study.py --mode all
```

Results are saved under:

```text
output/convergence/
```

Use `--no-phase-change` for a pure heat-conduction control calculation with
constant amorphous GST properties.

## MATLAB export

The crystallinity profile can be saved/read as `.mat`:

```powershell
.\.venv\Scripts\python.exe convert_crystallinity_npz_to_mat.py
```

Open in MATLAB with:

```matlab
open_crystallinity_profile_matlab
```

## Git notes

The repository intentionally ignores generated and heavy files:

```gitignore
.venv/
.matplotlib_cache/
__pycache__/
lg_modes_test/
*.zip
```

The `output/` folder is not ignored, so selected generated data and figures can
be committed when they are useful for reproducing diagnostics, for example:

```powershell
git add output/wpd_datasets.csv
git add output/fluence_fit_from_image/fluence_fit.png
```

Large temporary folders, virtual environments, Python caches, `lg_modes_test/`,
and zip archives stay out of Git.

## Numerical notes

The article reports a very fine mesh; the default scripts use coarser but more
practical grids. The femtosecond pulse is applied as an instantaneous enthalpy
increment because thermal diffusion during the pulse is negligible; subsequent
cooling uses implicit finite-volume time integration.

The exact non-isothermal crystallinity assignment is not fully specified in the
paper. The WPD-TTT path uses the article's graphical idea: a cooling curve is
assigned the crystalline fraction of the TTT contour it reaches/intersects.

For new substrates or very thin films, validate against experiment or COMSOL and
consider adding a GST/substrate thermal boundary resistance through
`interface_resistance`.
