# Upstream provenance and experiment map

## Source snapshot

- Upstream: <https://github.com/wushulin/ActaPinT>
- Branch inspected: `main`
- Pinned commit: `68431b7`
- Companion article: <https://doi.org/10.1017/S0962492924000072>
- Upstream language: MATLAB
- Python conversion location: `actapint/`

The upstream snapshot contains 24 MATLAB files, all of them ported. They are
retained at repository root so numerical choices can be checked against the
original scripts, which matters more often than it should: several published
figures are only reproduced by following a script against its own paper, and a
few by following the paper against its script. Each such case is recorded in the
per-section reproduction notes and summarized under *Known source corrections*.

## Implemented mapping

### Section 2 — model problems

| Python entry point | Upstream MATLAB source | Paper coverage |
|---|---|---|
| `figure_2_1` | `Solution_Heat_ADE.m` (`a=0`), `g.m` | Figure 2.1(a–d), all 4 panels |
| `figure_2_2` | `Solution_Heat_ADE.m` (`a=1`), `g.m` | Figure 2.2(a–h), all 8 panels |
| `figure_2_3` | `Solution_Bergers.m`, `g.m` | Figure 2.3(a–h), all 8 panels |
| `figure_2_4` | `Solution_Wave.m`, `g.m` | Figure 2.4(a–d), all 4 panels |
| `section2` | the four scripts above | every Section 2 panel in one run |

### The schematics — Figures 1.1, 3.2, 3.4, 3.7, 4.1 and 4.7

No script ships for any of them, and three of the six are computed rather than
drawn; see
[`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md#the-six-schematics).

| Python entry point | Upstream MATLAB source | Paper coverage |
|---|---|---|
| `figure_1_1` | no upstream script | Figure 1.1, forward Euler on the time axis |
| `figure_3_2` | no upstream script | Figure 3.2, red–black SWR tents |
| `figure_3_4` | no upstream script | Figure 3.4, the PIDC pipeline, 6 panels |
| `figure_3_7` | no upstream script | Figure 3.7, the two ParaExp steps |
| `figure_4_1` | no upstream script | Figure 4.1, the two time grids |
| `figure_4_7` | no upstream script | Figure 4.7, FCF-relaxation |
| `schematics` | none | all six in one run |

### Section 3 — PinT methods for hyperbolic problems

| Python entry point | Upstream MATLAB source | Paper coverage |
|---|---|---|
| `figure_3_1` | no upstream script | Figure 3.1(a–b), OSWR theory and four-subdomain counts |
| `figure_3_3` | `UTP_DD_Method.m`, `RedBlackSubdomains.m` | Figure 3.3, 4 panels |
| `figure_3_5` | `IDC_PIDC_ADE.m` | Figure 3.5, 4 panels |
| `figure_3_6` | no upstream script | Figure 3.6, 4 panels |
| `figure_3_8` | `ParaExp_Burgers.m` | Figure 3.8, 3 panels |
| `figure_3_9`, `figure_3_10` | `Direct_ParaDiag_ADE.m` | Figures 3.9, 3.10 |
| `figure_3_11`, `table_3_1` | `Direct_ParaDiag_Wave.m` | Figure 3.11, Table 3.1 |
| `figure_3_12` | no upstream script | Figure 3.12, 2 panels |
| `figure_3_13`, `table_3_2` | `Direct_ParaDiag_Burgers.m` | Figure 3.13, Table 3.2 |
| `figure_3_14` | no upstream script | Figure 3.14 |
| `figure_3_15`, `figure_3_16` | `Iterative_ParaDiag_ADE.m`, `Iterative_ParaDiag_Wave.m` | Figures 3.15, 3.16 |
| `figure_3_17` | `Iterative_ParaDiag_Wave.m` | Figure 3.17, 6 panels |
| `figure_3_18` | `Iterative_ParaDiag_RoundError.m` | Figure 3.18, 2 panels |
| `section3` | the scripts above | every Section 3 experiment in one run |

### Section 4 — PinT methods for parabolic problems

| Python entry point | Upstream MATLAB source | Paper coverage |
|---|---|---|
| `figure_4_2`, `figure_4_3`, `figure_4_4` | `Parareal_Heat_ADE.m` | Figures 4.2, 4.3, 4.4 |
| `figure_4_5` | `Parareal_Heat_ADE.m`, `Parareal_Burgers.m` | Figure 4.5(a–b) |
| `figure_4_6` | no upstream script | Figure 4.6, PFASST |
| `figure_4_8` | no upstream script | Figure 4.8, 2 panels |
| `figure_4_9`, `figure_4_10` | `MGRiT_Heat_ADE.m` | Figures 4.9, 4.10 |
| `figure_4_11` | `MGRiT_Burgers.m` | Figure 4.11, 3 panels |
| `figure_4_12` | `PCGC_Parareal_Heat_ADE.m` | Figure 4.12, 2 panels |
| `figure_4_13` | `PCGC_Parareal_Burgers.m` | Figure 4.13, 2 panels |
| `figure_4_14`, `figure_4_15`, `figure_4_16` | `DiagG_Parareal_Heat_Wave.m` | Figures 4.14, 4.15, 4.16 |
| `figure_4_17` | no upstream script | Figure 4.17, 2 panels |
| `figure_4_18` | no upstream script | Figure 4.18, 3 panels |
| `figure_4_19`, `figure_4_20`, `figure_4_21` | `STMG_Heat_ADE.m` | Figures 4.19, 4.20, 4.21 |
| `figure_4_22` | `STMG_Bergers.m` | Figure 4.22 |
| `section4` | the scripts above | every Section 4 experiment in one run |

Table 4.1 is not in this map: it reports weak and strong scaling of a full STMG
method to 262 144 cores, quoted from Gander and Neumüller (2016) rather than
produced by this paper's code.

### Script-faithful entry points

These run the committed script defaults rather than a published figure, and are
kept because they are the cheapest way to compare a port against its original:

| Python entry point | Upstream MATLAB source | What it produces |
|---|---|---|
| `solution_heat_ade` | `Solution_Heat_ADE.m` | the script defaults, i.e. Figure 2.2(d) alone |
| `solution_burgers` | `Solution_Bergers.m` | the script defaults, i.e. Figure 2.3(d) alone |
| `solution_wave` | `Solution_Wave.m` | source-free Dirichlet wave; no published panel uses this combination |
| `parareal_heat_ade` | `Parareal_Heat_ADE.m` | the linear case of Figure 4.5 |
| `parareal_burgers` | `Parareal_Burgers.m` | the nonlinear case of Figure 4.5 |
| `mgrit_heat_ade` | `MGRiT_Heat_ADE.m` | Figures 4.9–4.10 as the script draws them |
| `stmg_heat_ade` | `STMG_Heat_ADE.m` | Figures 4.19–4.20 as the script draws them |
| `iterative_paradiag_ade` | `Iterative_ParaDiag_ADE.m` | Figure 3.15(c,d) |
| `paper_validation` wave case | `Iterative_ParaDiag_Wave.m`, adjusted to the paper parameters | Figure 3.15(e,f) |

## Known source corrections

Scripts that cannot run as committed:

`MGRiT_Heat_ADE.m` contains `nu=0.002max;`, which is not valid MATLAB. The
Python port uses `nu=0.002`, matching the corresponding branch, title, and
parameter sweep in the paper.

`Solution_Wave.m` sets `A2(1,1)=nu/dx^2` in its Neumann branch, but `nu` is never
assigned in that script, so the branch cannot run as committed. The port uses
`c^2/dx^2`, the coefficient used by the surrounding branches.

Places where the script and the published figure disagree, resolved in favour of
the figure:

`Solution_Bergers.m` builds the flux operator as `.../(4*dx)`, that is
`0.5*(u^2)_x`, matching equation (2.6). The published panels of Figure 2.3 are
only reproduced with `.../(2*dx)`, and so are Figures 4.5(b) and 4.11. Figures
4.13 and 4.22 are reproduced with the committed factor. See
[`docs/SECTION2_REPRODUCTION.md`](docs/SECTION2_REPRODUCTION.md) for the
measurements and the alternatives that were ruled out.

`DiagG_Parareal_Heat_Wave.m` omits the factor `1 - alpha` that equation (4.28)
puts in front of the initial condition of the coupled window system. The wave
panels of Figures 4.15 and 4.16 are reproduced as committed; the heat panels of
Figure 4.14 need the factor, without which the defect `F - F*` is largest exactly
on the modes the fine solver leaves untouched and the curves bend.

`STMG_Bergers.m` sets `soomth_iter_num=4` and `eeta=[0.25]`, then overwrites `eta`
with `0.28` inside the iteration loop. Figure 4.22 is captioned with two smoothing
steps, and two is what reproduces it; the damping parameter that reproduces it is
the overwritten `0.28`.

Quirks that are kept because the published figures depend on them:

`STMG_Heat_ADE.m` computes the residual that drives the coarse grid correction
from `uold`, the iterate before the last smoothing step, rather than from the
smoothed one, because it reuses the loop variable of its block Jacobi sweep.

`STMG_Bergers.m` freezes the previous coarse level at the restricted iterate
inside `solve_Kc`, so its coarse problem decouples into one nonlinear equation
per coarse level instead of being a sequential solve, and its first coarse level
carries no initial condition.

The Parareal and MGRiT scripts return the last Newton iterate when the iteration
has not converged within the cap (1000 iterations in `Pro_IE` of the Parareal
scripts, 50 in the MGRiT ones) instead of failing, which is what lets the
divergent panels of Figures 4.5(b) and 4.11 run to the end.
