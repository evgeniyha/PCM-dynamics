# GST225 laser crystallization model

Python implementation of the thermal and crystallization model from:

T. Kunkel et al., *Crystallization of GST225 thin film induced by a single
femtosecond laser pulse: Experimental and theoretical study*, Materials
Science in Semiconductor Processing 139 (2022) 106350.

## What is implemented

- 2D axisymmetric heat conduction in a GST film and homogeneous substrate;
- Gaussian 1030 nm pulse and Beer-Lambert absorption;
- optional melting/freezing latent heat through effective heat capacity;
- transient homogeneous nucleation, growth and isothermal TTT curves;
- conversion of cooling rate to crystallinity using the tangent construction
  employed in the paper.
- a two-dimensional crystalline-fraction profile `Xc(r,z)` analogous to Fig. 5.

The default absorption uses `alpha_linear = 10 um^-1` in the heat-source
prefactor and `alpha_eff = 16.2 um^-1` in the Beer-Lambert attenuation term,
following the paper's interpretation `alpha_eff = alpha + alpha_f`. The
Gaussian fluence is normalised as
`F(r) = 2 E / (pi w^2) exp(-2 r^2 / w^2)`, where `w` is the 1/e^2 fluence
radius. This integrates to the pulse energy `E` and matches the fluence peak
and width in Fig. 4a; the printed Eq. (5) in the article appears to omit the
radial factor of two in the exponent. The
fragility `m = 67` and interfacial energy `sigma = 0.063 J/m^2` follow the
paper. Predictions for a new film or substrate should therefore be validated
experimentally. For very thin films, set a measured GST/substrate thermal
boundary resistance with `interface_resistance`.

## Run

```powershell
python run_article_case.py
python run_article_case.py --film-nm 50 --substrate tungsten --energy-nj 650
python run_article_case.py --no-phase-change
python run_article_case.py --crystallinity-method path_intersection
```

The plot is written to `output/article_case.png`.

The default crystallinity profile uses the article-style TTT/cooling-curve
intersection construction. A cooling-rate diagnostic approximation is available
with `--crystallinity-method cooling_rate` for comparison.

## Optional convergence study

The convergence study is never started automatically. A short workflow check:

```powershell
.\.venv\Scripts\python.exe convergence_study.py --quick
```

Full spatial, temporal and domain-size comparison (potentially expensive):

```powershell
.\.venv\Scripts\python.exe convergence_study.py --mode all
```

The convergence study compares only temperature metrics. Add
`--no-phase-change` to run a pure heat-conduction control calculation with
constant amorphous GST properties. Results are saved under
`output/convergence/` as CSV, JSON and PNG files.

Required packages: NumPy, SciPy and Matplotlib. Run tests with:

```powershell
python -m unittest -v
```

## Numerical fidelity

The example uses a coarser spacing than the article so that it runs on a normal
workstation. Change `Grid(dr=50e-9, dz_film=5e-9, dz_substrate=5e-9)` to
reproduce the reported film mesh and extend it into the substrate. This can
require a very large amount of memory. The 185 fs heating
stage is applied as an instantaneous enthalpy increment because thermal
diffusion during the pulse is negligible; subsequent cooling uses implicit
finite-volume time integration with the paper's 0.5 ns default step.

The article does not supply source code or every numerical convention. In
particular, the exact crystallinity assignment for a non-isothermal trace is
reconstructed from its stated TTT tangent procedure. This implementation is a
transparent research reproduction, not the authors' original program.

With the table value `sigma = 0.063 J/m^2` and the printed value `m = 67`, the
implementation gives a TTT nose near 670 K but slower absolute TTT times than
reported in the paper. This indicates that at least one numerical/material
convention is absent from the publication. Do not hide that discrepancy by
treating the code as an exact replica; recalibrate kinetic parameters against
your own experiment.
