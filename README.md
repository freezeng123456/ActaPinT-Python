# ActaPinT-Python

[![tests](https://github.com/freezeng123456/ActaPinT-Python/actions/workflows/tests.yml/badge.svg)](https://github.com/freezeng123456/ActaPinT-Python/actions/workflows/tests.yml)

Python conversion, verification, and extension of the numerical experiments in
[`wushulin/ActaPinT`](https://github.com/wushulin/ActaPinT), the companion
MATLAB repository for:

> Martin J. Gander, Shu-Lin Wu, and Tao Zhou, “Time parallelization for
> hyperbolic and parabolic problems,” *Acta Numerica* (2025).
> <https://doi.org/10.1017/S0962492924000072>

The original MATLAB files in this repository are pinned to upstream commit
[`68431b7`](https://github.com/wushulin/ActaPinT/commit/68431b7). The Python
package is a research reproduction: it makes the main experiments callable,
records numerical metrics as JSON, and creates publication-ready figures in
both SVG and PNG formats.

## Scope

The current implementation covers the three model equations and four main
time-parallel algorithm families:

- heat/advection-diffusion, viscous Burgers, and wave equations;
- Parareal and two-level MGRiT;
- iterative ParaDiag with FFT-based circulant preconditioning;
- two-grid space-time multigrid (STMG).

Section 2 is reproduced in full: all 24 space-time panels of Figures 2.1–2.4,
checked pixel by pixel against the published rasters rather than by eye. See
[`docs/SECTION2_REPRODUCTION.md`](docs/SECTION2_REPRODUCTION.md).

Section 3 has implementations for all fifteen numerical figures and both tables.
Figure 3.14 remains a partial reproduction: the NKA acceleration at `nu = 0.02`
does not match the published curves. The other comparisons, including the
normalization and roundoff differences, are recorded per figure. The three
schematics are drawn too, each of them from a computation. That covers
Schwarz waveform relaxation, pipelined and revisionist integral deferred
correction, ParaExp, and both ParaDiag branches, including the geometric-mesh and
boundary-value-method variants of ParaDiag I and the quasi-Newton treatment of
Burgers' equation. See
[`docs/SECTION3_REPRODUCTION.md`](docs/SECTION3_REPRODUCTION.md).

Section 4 has implementations for all twenty figures that carry numerical
content, Figures 4.2 through 4.22, and the two schematics are drawn as well. That covers
Parareal and its two convergence regimes, PFASST, MGRiT with FCF-relaxation, the
two ParaDiag-based Parareal variants — one diagonalizing the coarse grid
correction, one the coarse solver — and space-time multigrid, for the heat
equation, the advection-diffusion equation, Burgers' equation and the wave
equation. The 3D P1-FEM / dG(3) multigrid baseline behind Table 4.1 is now
implemented at the original spatial problem size. The local 2/4/8-step cases
converge in seven iterations; the historical supercomputer timings remain
separate from these serial numerical checks. See
[`docs/SECTION4_REPRODUCTION.md`](docs/SECTION4_REPRODUCTION.md).

The older paper-focused suite validates Figures 3.15, 4.5, 4.9–4.10 and
4.19–4.20; the per-section groups above supersede it. See
[`UPSTREAM.md`](UPSTREAM.md) for the source mapping and
[`PAPER_VALIDATION_REPORT.md`](PAPER_VALIDATION_REPORT.md) for its quantitative
comparison.

Chapters 1 and 5 carry no experiments: Figure 1.1 is a schematic of the forward
Euler time axis, drawn from the marker heights of the published page, and Chapter 5
is a one-page conclusion. The six schematics of the article run as one group,
`actapint schematics`, and what each of them computes is described in
[`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md#the-six-schematics). Coverage of all 45
figures and 3 tables, the cost of a full reproduction, and the register of
recovered parameters and corrections are in
[`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md); the committed plots and
machine-readable outputs are collected in [`docs/RESULTS.md`](docs/RESULTS.md).

The [2026-11-10 reproduction checks](docs/REPRODUCTION_CHECKS_2026-11-10.md)
add independent NKA solves, the 3D STMG baseline, and portable numerical tests.

The [2026-11-10 reproduction checks](docs/REPRODUCTION_CHECKS_2026-11-10.md)
add independent NKA solves, the 3D STMG baseline, and portable numerical tests.

## Reproduced paper results

The paper-scale suite was rerun with Python 3.11 on a clean Linux environment:

| Paper figure | Reproduced result |
|---|---|
| Figures 2.1–2.4 | All 24 panels; median smoothed correlation with the published rasters 0.996, median decile mismatch 0.0026 |
| Figures 3.1–3.18, Tables 3.1–3.2 | All fifteen numerical figures and both tables; every OSWR iteration count of Figure 3.1, all twelve condition numbers of Table 3.1, the sequential column of Table 3.2 exactly, and the ParaDiag II spectra to three digits |
| Figure 3.15 | Heat: 2 GMRES iterations; ADE: 3 at $\nu=10^{-3}$ and 13 at $\nu=10^{-6}$; wave: 89 iterations to preconditioned residual $10^{-11}$ |
| Figure 4.5 | ADE: 14, 24, 35 iterations; Burgers: 14, 21, 25 for $\nu=1,0.1,0.02$ |
| Figures 4.9–4.10 | Long-time factors agree to the displayed precision, including $1.4211/1.2812$ for Parareal/MGRiT at $\nu=0.002$; the 124 error markers read off Figure 4.10 are reproduced to 6% |
| Figures 4.19–4.20 | Best sampled damping $\eta=0.500$ for heat and $\eta=0.372$ for ADE at $\nu=0.01$ |

The wave run takes 103 SciPy iterations to satisfy its true-residual stopping
test, while its preconditioned residual first falls below $10^{-11}$ at
iteration 89. The paper curve ends near iteration 88; this is consistent at the
level expected when MATLAB and SciPy use different GMRES residual and stopping
conventions.

## Installation

Python 3.10 or newer is supported. The 2026-11-10 numerical stack uses Python
3.12 with `constraints-reproduction.txt`; use `pip install -c
constraints-reproduction.txt -e ".[test]"` to select those versions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Reproduce

Run a fast smoke test:

```bash
actapint paper_validation --quick --output-dir results/quick
```

Run the paper-scale validation:

```bash
actapint paper_validation --output-dir results/paper
```

Reproduce every Section 2 panel, about 30 seconds on one core:

```bash
actapint section2 --output-dir results/section2
```

Individual figures are available as `figure_2_1` through `figure_2_4`. To check
them against the article rather than by eye:

```bash
python scripts/compare_with_paper.py paper.pdf \
  --output docs/data/section2_paper_comparison.json
```

Reproduce every Section 3 experiment, 10 minutes on one core:

```bash
actapint section3 --output-dir results/section3
```

Individual experiments are available under their figure names, `figure_3_1`,
`figure_3_3`, `figure_3_5`, `figure_3_6`, `figure_3_8` through `figure_3_18`,
`table_3_1` and `table_3_2`.

Reproduce every Section 4 experiment, 25 minutes on one core:

```bash
actapint section4 --output-dir results/section4
```

Individual experiments are available as `figure_4_2` through `figure_4_6` and
`figure_4_8` through `figure_4_22`.

Draw the six schematics, Figures 1.1, 3.2, 3.4, 3.7, 4.1 and 4.7, in 4 seconds:

```bash
actapint schematics --output-dir results/schematics
```

They are available individually as `figure_1_1`, `figure_3_2`, `figure_3_4`,
`figure_3_7`, `figure_4_1` and `figure_4_7`.

On a CUDA 12 server, install the optional CuPy backend and batch the independent
Burgers fine propagators on the GPU:

```bash
python -m pip install -e ".[gpu,test]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  actapint paper_validation --backend gpu \
  --output-dir results/paper-gpu
```

`--backend auto` is the default: it selects the GPU when CUDA-capable CuPy is
available and otherwise uses the CPU. The implementation is hybrid. It keeps
the causal coarse Parareal sweep and the other paper experiments on the CPU,
while one GPU batch advances all independent Burgers time slices. See
[`docs/GPU_ACCELERATION.md`](docs/GPU_ACCELERATION.md) for measured speedups,
correctness checks, and the remaining optimization opportunities.

Run the original-size 3D STMG baseline and the 24-case independent NKA audit:

```bash
actapint stmg_fem_3d --output-dir results/checks
actapint figure_3_14_audit --output-dir results/checks
```

These checks write JSON; `--quick` selects smaller verification problems.

Run the original-size 3D STMG baseline and the 24-case independent NKA audit:

```bash
actapint stmg_fem_3d --output-dir results/checks
actapint figure_3_14_audit --output-dir results/checks
```

These checks write JSON; `--quick` selects smaller verification problems.

Run every converted experiment:

```bash
actapint all --output-dir results/full
```

Every plotting experiment writes an editable SVG, a high-resolution PNG, and a
JSON metrics file. Random initial states use fixed NumPy seeds.

## Numerical notes

- `MGRiT_Heat_ADE.m` contains the invalid statement `nu=0.002max;`; the Python
  implementation uses `nu=0.002`, consistent with the paper and surrounding
  branches.
- `Solution_Bergers.m` and equation (2.6) both specify the flux `0.5*(u^2)_x`,
  but the published panels of Figure 2.3 are only reproduced with `(u^2)_x`. Grid
  refinement, upwinding, extra diffusion and a source amplitude change were ruled
  out first; see
  [`docs/SECTION2_REPRODUCTION.md`](docs/SECTION2_REPRODUCTION.md).
- `Solution_Wave.m` references an undefined `nu` in its Neumann branch, so that
  branch cannot run as committed; the port uses `c^2`.
- `SWR_for_ADE.m` scales a centred first difference by `dt/dx` rather than
  `dt/(2*dx)`, so it transports at speed 2; Figure 3.1(b) is reproduced by the
  consistent coefficient. The same script computes the Robin parameter as
  `p_tilde*nu/l`, not the `p_tilde*s/nu` printed in Theorem 3.1, and only its
  version reproduces the figure; see
  [`docs/SECTION3_REPRODUCTION.md`](docs/SECTION3_REPRODUCTION.md).
- `Direct_ParaDiag_ADE.m` commits its advection coefficient as `adc = 0`, which
  selects the heat equation, so the coefficient behind the published
  advection-diffusion panels of Figures 3.9 and 3.10 is absent from the
  repository. It is `1/2`; the unit speed of equation (2.5) puts those panels a
  factor of four high.
- The mesh ratio `rho_num` of Figure 3.10, Figure 3.11(b) and Table 3.1 is a
  single value shared by all step counts, not the per-step-count minimizer the
  text defines. With the minimizer neither the error growth of Figure 3.10 nor
  the condition numbers of Table 3.1 appear.
- `DiagG_Parareal_Heat_Wave.m` omits the factor `1-alpha` that equation (4.28)
  puts in front of the initial condition of the coupled window system. The wave
  panels of Figures 4.15 and 4.16 are reproduced as committed, the heat panels of
  Figure 4.14 only with the factor; see
  [`docs/SECTION4_REPRODUCTION.md`](docs/SECTION4_REPRODUCTION.md).
- `STMG_Bergers.m` commits four smoothing steps and then overwrites its damping
  parameter with `0.28` inside the iteration; Figure 4.22 is reproduced with the
  two smoothing steps of its caption and that overwritten damping.
- Neither the text nor the scripts state the three viscosities of Figure 4.11 or
  the fine solver of Figure 4.12; they were recovered by matching the published
  panels, and every such recovery is listed in
  [`docs/SECTION4_REPRODUCTION.md`](docs/SECTION4_REPRODUCTION.md).
- `Iterative_ParaDiag_ADE.m` integrates with `theta = 1`, backward Euler, where
  the text and the caption of Figure 3.16 both say trapezoidal rule.
- `Direct_ParaDiag_ADE.m` and `Direct_ParaDiag_Wave.m` apply `It/V` as a formed
  matrix rather than solving with `V`, and `Iterative_ParaDiag_RoundError.m`
  solves the blocks of `P_alpha` rather than inverting them once. Both are less
  accurate than the obvious alternative and both are needed: they are what puts
  the blow-up of Figures 3.10 to 3.12 and the accuracy floors of Figure 3.18
  where the paper shows them.
- MATLAB and NumPy use different random-number generators. Initial errors need
  not match bit-for-bit, while convergence factors, trends, and final states are
  directly comparable.
- These examples are dominated by sparse CPU linear algebra and FFTs. A GPU is
  not required; the optional backend accelerates the independent nonlinear fine
  propagators that dominate the paper suite.

## Rights and attribution

No license file was present in the inspected upstream commit. The included
MATLAB files therefore remain subject to the upstream authors’ terms. This
repository documents provenance and is intended for research reproduction; it
does not relicense upstream material.
