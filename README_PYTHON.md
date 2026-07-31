# Legacy conversion notes

The maintained project overview is now [`README.md`](README.md). This file
preserves the notes produced during the original Python conversion on the old
server.

This directory adds a runnable Python implementation for the main numerical
experiments in the original MATLAB-only repository.

## What was found

The repository has no README, test data, paper metadata, environment file, or
stored reference output. It contains standalone MATLAB scripts for:

- reference solutions of advection-diffusion/heat, Burgers, and wave equations;
- Parareal and MGRiT;
- direct and iterative ParaDiag;
- space-time multigrid (STMG);
- diagonal coarse-grid correction, ParaExp, SWR, IDC/PIDC, and a waveform
  domain-decomposition experiment.

`MGRiT_Heat_ADE.m` contains the invalid statement `nu=0.002max;`. The Python
port uses `nu=0.002`, which is the only value consistent with the surrounding
title, branches, and parameter scale.

## Implemented runnable experiments

| Python experiment | Original MATLAB source |
|---|---|
| `solution_heat_ade` | `Solution_Heat_ADE.m` |
| `solution_burgers` | `Solution_Bergers.m` |
| `solution_wave` | `Solution_Wave.m` |
| `parareal_heat_ade` | `Parareal_Heat_ADE.m` |
| `parareal_burgers` | `Parareal_Burgers.m` |
| `mgrit_heat_ade` | `MGRiT_Heat_ADE.m` |
| `iterative_paradiag_ade` | `Iterative_ParaDiag_ADE.m` |
| `stmg_heat_ade` | `STMG_Heat_ADE.m` damping/smoother experiment |

The shared implementation is in `actapint/`, avoiding the global variables and
duplicated local functions used by the MATLAB scripts.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_experiments.py all --quick --output-dir results/quick
python3 run_experiments.py parareal_heat_ade --output-dir results/full
python3 run_experiments.py paper_validation --output-dir results/paper
```

Each run writes a PNG figure and a JSON metrics file. `--quick` preserves the
algorithm and parameter ratios while reducing grid sizes for smoke testing.

`paper_validation` reproduces the parameter studies behind Figures 3.15, 4.5,
4.9-4.10 and 4.19-4.20 of the paper. Run it without `--quick` for the paper grid
sizes.

## Reproducibility limits

MATLAB's `rng('default')` and NumPy use different random-number generators, so
experiments initialized with random all-at-once states do not have identical
first-iteration errors. The converged numerical solution and convergence trend
are the meaningful comparison. Sparse LU/FFT ordering also causes normal
floating-point differences near machine precision.
