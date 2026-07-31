# ActaPinT-Python

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
  not required by the current implementation.

## Rights and attribution

No license file was present in the inspected upstream commit. The included
MATLAB files therefore remain subject to the upstream authors’ terms. This
repository documents provenance and is intended for research reproduction; it
does not relicense upstream material.
