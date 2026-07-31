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

The paper-focused suite validates Figures 3.15, 4.5, 4.9–4.10, and 4.19–4.20.
See [`UPSTREAM.md`](UPSTREAM.md) for the source mapping and
[`PAPER_VALIDATION_REPORT.md`](PAPER_VALIDATION_REPORT.md) for the current
quantitative comparison.

The committed plots and machine-readable outputs are collected in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Reproduced paper results

The paper-scale suite was rerun with Python 3.11 on a clean Linux environment:

| Paper figure | Reproduced result |
|---|---|
| Figure 3.15 | Heat: 2 GMRES iterations; ADE: 3 at $\nu=10^{-3}$ and 13 at $\nu=10^{-6}$; wave: 89 iterations to preconditioned residual $10^{-11}$ |
| Figure 4.5 | ADE: 14, 24, 35 iterations; Burgers: 14, 21, 25 for $\nu=1,0.1,0.02$ |
| Figures 4.9–4.10 | Long-time factors agree to the displayed precision, including $1.4211/1.2812$ for Parareal/MGRiT at $\nu=0.002$ |
| Figures 4.19–4.20 | Best sampled damping $\eta=0.500$ for heat and $\eta=0.372$ for ADE at $\nu=0.01$ |

The wave run takes 103 SciPy iterations to satisfy its true-residual stopping
test, while its preconditioned residual first falls below $10^{-11}$ at
iteration 89. The paper curve ends near iteration 88; this is consistent at the
level expected when MATLAB and SciPy use different GMRES residual and stopping
conventions.

## Installation

Python 3.10 or newer is recommended.

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
