# Reproducing the whole article

This is the index of the reproduction: what the article contains, what of it has
been reproduced, what it costs to run, and every place where the reproduction had
to depart from what the article or its scripts say. The three chapters that carry
experiments have their own reports, and this one does not repeat them:

* [`SECTION2_REPRODUCTION.md`](SECTION2_REPRODUCTION.md) — model problems
* [`SECTION3_REPRODUCTION.md`](SECTION3_REPRODUCTION.md) — PinT for hyperbolic problems
* [`SECTION4_REPRODUCTION.md`](SECTION4_REPRODUCTION.md) — PinT for parabolic problems

The article is M. J. Gander, S.-L. Wu and T. Zhou, *Time parallelization for
hyperbolic and parabolic problems*, Acta Numerica 34 (2025), 385–489,
[doi:10.1017/S0962492924000072](https://doi.org/10.1017/S0962492924000072). Its
companion code is <https://github.com/wushulin/ActaPinT>, pinned here at commit
`68431b7`: 24 MATLAB scripts, all ported.

*Reproduced* means the experiment was rebuilt from the script or, where no script
ships, from the text, and then compared against the published evidence at the
finest granularity the figure allows: printed values where the article prints them,
markers extracted from the published raster where it does not, and pixel statistics
for the space–time colour panels of Chapter 2. Each experiment writes an editable
SVG, a PNG and a JSON file of the plotted numbers; the JSON files are committed
under [`data`](data) so the comparisons can be redone without rerunning anything.

## Coverage

The article has 45 figures and 3 tables. Thirty-nine figures carry measurements
and have runnable implementations; Figure 3.14 remains a partial numerical
match. The six schematics carry none and are redrawn, three of them
from a computation rather than by hand ([below](#the-six-schematics)). Of the
three tables, two are reproduced. The third now has an original-size spatial
numerical baseline; its supercomputer scaling measurements are separate. See
[the 2026-11-10 checks](REPRODUCTION_CHECKS_2026-11-10.md).

| Chapter | Figures | Schematic | Numerical | Tables | Reproduced |
|---|---:|---:|---:|---:|---:|
| 1 Introduction | 1 | 1 | 0 | 0 | 1 figure |
| 2 Model problems | 4 | 0 | 4 | 0 | 4 of 4 |
| 3 Hyperbolic problems | 18 | 3 | 15 | 2 | 18 implemented figures (3.14 partial), 2 tables |
| 4 Parabolic problems | 22 | 2 | 20 | 1 | 22 figures, Table 4.1 numerical and MPI baseline |
| 5 Conclusions | 0 | 0 | 0 | 0 | prose only |

### Chapter 1 — Introduction

| Item | Content | Status |
|---|---|---|
| Figure 1.1 | the time axis of forward Euler, showing its sequential nature | drawn, from the marker heights of the page |

### Chapter 2 — Model problems linking the parabolic and hyperbolic world

Four figures, 24 space–time colour panels, all reproduced and compared to the
published rasters pixel by pixel; the median smoothed correlation is 0.995 or
better in every figure.

| Item | Content | Upstream script | Status |
|---|---|---|---|
| Figure 2.1 | heat equation, Dirichlet / Neumann / periodic, 4 panels | `Solution_Heat_ADE.m`, `g.m` | reproduced |
| Figure 2.2 | advection–diffusion for weakening diffusion, 8 panels | `Solution_Heat_ADE.m`, `g.m` | reproduced |
| Figure 2.3 | Burgers' equation and shock formation, 8 panels | `Solution_Bergers.m`, `g.m` | reproduced with flux `(u^2)_x` |
| Figure 2.4 | second-order wave equation, 4 panels | `Solution_Wave.m`, `g.m` | reproduced |

### Chapter 3 — Effective PinT methods for hyperbolic problems

| Item | Content | Upstream script | Status |
|---|---|---|---|
| Figure 3.1 | OSWR factor and four-subdomain counts, 2 panels | `SWR_for_ADE.m` | reproduced, every marker |
| Figure 3.2 | red–black SWR tents | — | drawn, exact region propagated rather than sketched |
| Figure 3.3 | red–black SWR / UTP error fields, 4 panels | `UTP_DD_Method.m`, `RedBlackSubdomains.m` | reproduced |
| Figure 3.4 | the PIDC pipeline schedule | — | drawn, sweeps of (3.11) computed |
| Figure 3.5 | IDC against pipelined IDC, 4 panels | `IDC_PIDC_ADE.m` | reproduced, one constant per rough panel |
| Figure 3.6 | IDC against revisionist IDC, 4 panels | text only | reproduced, one constant per rough panel |
| Figure 3.7 | the two ParaExp steps | — | drawn, (3.13) to (3.15) computed |
| Figure 3.8 | ParaExp against Parareal for Burgers, 3 panels | `ParaExp_Burgers.m` | reproduced |
| Figure 3.9 | error against the geometric-mesh parameter, 2 panels | `Direct_ParaDiag_ADE.m` | reproduced, advection coefficient recovered |
| Figure 3.10 | blow-up past the roundoff threshold | `Direct_ParaDiag_ADE.m` | reproduced with the ratio held fixed |
| Figure 3.11 | wave equation, geometric and uniform steps, 2 panels | `Direct_ParaDiag_Wave.m` | reproduced, both panels |
| Table 3.1 | condition number of the eigenvector matrix | `Direct_ParaDiag_Wave.m` | reproduced, all twelve entries |
| Figure 3.12 | geometric steps against the BVM discretization, 2 panels | text only | reproduced, both panels marker for marker |
| Figure 3.13 | BVM ParaDiag I for Burgers | `Direct_ParaDiag_Burgers.m` | reproduced to one iteration |
| Figure 3.14 | two quasi-Newton variants for Burgers | text only | averaged trend matched; NKA acceleration at `nu = 0.02` remains unmatched |
| Table 3.2 | Jacobian solves, sequential against parallel | `Direct_ParaDiag_Burgers.m` | reproduced, sequential column exactly |
| Figure 3.15 | preconditioned spectra and GMRES, 6 panels | `Iterative_ParaDiag_ADE.m`, `Iterative_ParaDiag_Wave.m` | reproduced, spectra to 3 digits |
| Figure 3.16 | head–tail coupled waveform relaxation, 2 panels | `Iterative_ParaDiag_ADE.m` | reproduced, every marker |
| Figure 3.17 | eigenvalues of the iteration matrix, 6 panels | `Iterative_ParaDiag_Wave.m` | reproduced |
| Figure 3.18 | the same iteration in two implementations, 2 panels | `Iterative_ParaDiag_RoundError.m` | reproduced |

### Chapter 4 — PinT methods designed for parabolic problems

| Item | Content | Upstream script | Status |
|---|---|---|---|
| Figure 4.1 | the two-level time grid | — | drawn |
| Figure 4.2 | the two convergence regimes of Parareal, 2 panels | `Parareal_Heat_ADE.m` | reproduced, `J` recovered as 4 |
| Figure 4.3 | five fine solvers against one bound, 2 panels | `Parareal_Heat_ADE.m` | reproduced, every marker |
| Figure 4.4 | factor against the diffusion parameter | `Parareal_Heat_ADE.m` | reproduced, all four digits |
| Figure 4.5 | Parareal for heat, ADE and Burgers, 2 panels | `Parareal_Heat_ADE.m`, `Parareal_Burgers.m` | reproduced, published legend swapped |
| Figure 4.6 | PFASST for heat and ADE | text only | reproduced, all four factors within 0.6% |
| Figure 4.7 | FCF-relaxation | — | drawn |
| Figure 4.8 | regions where the factor is small, 2 panels | text only | reproduced |
| Figure 4.9 | MGRiT and Parareal factors, 6 panels | `MGRiT_Heat_ADE.m` | 5 of 6 to four digits, 1 printed value questioned |
| Figure 4.10 | measured error, MGRiT against Parareal, 4 panels | `MGRiT_Heat_ADE.m` | reproduced, every marker, two setups recovered |
| Figure 4.11 | MGRiT for Burgers, 3 panels | `MGRiT_Burgers.m` | reproduced, viscosities recovered |
| Figure 4.12 | diagonalized coarse grid correction, 2 panels | `PCGC_Parareal_Heat_ADE.m` | reproduced, factors to two digits |
| Figure 4.13 | the same for Burgers, 2 panels | `PCGC_Parareal_Burgers.m` | reproduced with the committed flux |
| Figure 4.14 | diagonalized coarse solver, heat, 2 panels | `DiagG_Parareal_Heat_Wave.m` | reproduced to 0.05 decades, `J = 20` recovered |
| Figure 4.15 | the same for the wave equation, 2 panels | `DiagG_Parareal_Heat_Wave.m` | reproduced, every point above `1e-10` to 0.1% |
| Figure 4.16 | sharpness of the wave-equation factor, 2 panels | `DiagG_Parareal_Heat_Wave.m` | reproduced |
| Figure 4.17 | iteration counts for Burgers, 2 panels | text only | reproduced to one iteration |
| Figure 4.18 | LFA factor over the high frequencies, 3 panels | text only | reproduced |
| Figure 4.19 | error against the damping parameter, 2 panels | `STMG_Heat_ADE.m` | reproduced |
| Figure 4.20 | STMG with one and three smoothing steps, 2 panels | `STMG_Heat_ADE.m` | reproduced, every marker |
| Figure 4.21 | the same sweep with the trapezoidal rule, 6 panels | `STMG_Heat_ADE.m` | reproduced |
| Figure 4.22 | STMG for Burgers | `STMG_Bergers.m` | reproduced, every marker |
| Table 4.1 | weak and strong scaling to 262 144 cores | — | 3D P1/dG(3) MPI baseline and current-machine scaling; historical timings retained |

### Chapter 5 — Conclusions

Chapter 5 is one page of prose with no figures, no tables and no experiments. Its
claims are about which method families survive which problem class, and the
evidence for each of them is the reproduced data of Chapters 3 and 4.

## The six schematics

Figures 1.1, 3.2, 3.4, 3.7, 4.1 and 4.7 print no measurement, so there is no
number to compare. What they do carry is content, and for three of them that
content is a computation, which is how they are drawn here rather than traced:

* **Figure 3.2**, red–black SWR with generous overlap. The blue region of each
  panel is where the method already has the exact solution, and it is propagated
  instead of sketched: on a grid with `dt = dx` the domain of dependence of the
  wave equation is the pair of neighbours one row below, so a point is exact
  exactly when both of those are, a physical boundary is exact throughout, and an
  artificial interface is exact only up to the time an earlier solve has made it
  so. The tents of panel (a) close at `T_1`, and the recursion then produces
  by itself the shape the caption of panel (c) calls a rhomboid: with the
  interfaces `x_1` and `x_3` exact up to `T_1`, the black solve on `(x_1, x_3)`
  is exact where `t <= x - x_1` and `t <= x_3 - x`, whose tip is `(x_2, T_2)`, and
  the union with the tents of the red solve is the quadrilateral
  `(x_2, 0), (x_1, T_1), (x_2, T_2), (x_3, T_1)`. Each solve advances the exact
  region by exactly one level, which the metrics record.
* **Figure 3.4**, the PIDC pipeline. The curves are real IDC sweeps of (3.11) with
  `theta = 1` on six windows of `u' = cos(3 t) - u`, six nodes each, so the sweep
  histories the caption describes converge the way the published ones do: every
  sweep of every window gains about an order of magnitude on the last, which is why
  the red curve of `k = 1` visibly misses the exact solution drawn with it and the
  one of `k = 4` sits on top of it.
* **Figure 3.7**, the two ParaExp steps. The red curves are the particular
  solutions `v_n` of (3.13), driven by a piecewise constant source from a fixed
  seed and starting from zero on each window, and the dashed curves are the
  homogeneous propagations `w_n` of (3.14). Every piece is marched on one shared
  grid, so the decomposition (3.15) can be checked rather than assumed: the sum of
  the drawn curves and the sequential solution of the same problem agree to
  `3e-14`.

The other three are geometry. Figure 1.1 is the one whose geometry carries
information, the shape of a forward Euler trajectory, and it is not invented here:
the thirteen markers of the published page sit `16.98` pt apart with heights that
are integer multiples of `1.869` pt above the time axis, that is at
`10, 12, 15, 17, 17, 15, 14, 13, 14, 15, 16, 17, 16` units, which is what the
reproduction draws. Figure 4.1 draws the two grids with the `J = 4` the page
shows, and Figure 4.7 the eight coarse points with `F` spanning two coarse
intervals from each of them, above the line from the even points and below it from
the odd ones. One detail of Figure 4.7 is worth recording: its caption says *the
dark circles represent the coarse time points*, but the vector art of the page
draws all eight identically, as unfilled rings — every drawn point is a coarse
point and the fine grid is not drawn at all.

```bash
actapint schematics --output-dir results/schematics
```

## Cost of a full reproduction

Measured on one core of the reference machine (Python 3.11 on Linux, all BLAS
threading disabled), with the whole article reproduced in four commands:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
actapint schematics --output-dir results/schematics
actapint section2 --output-dir results/section2
actapint section3 --output-dir results/section3
actapint section4 --output-dir results/section4
```

| Group | Experiments | Wall time | Peak memory | Where the time goes |
|---|---:|---:|---:|---|
| `schematics` | 6 | 4 s | 215 MiB | the characteristic propagation of Figure 3.2 |
| `section2` | 4 | 29 s | 633 MiB | the `dt = dx = 1e-3` space–time arrays of Figures 2.1 and 2.2 |
| `section3` | 17 | 10 min | 324 MiB | the quasi-Newton sweeps of Figures 3.13 and 3.14 |
| `section4` | 20 | 18 min | 235 MiB | the nonlinear experiments, 17 of the 18 minutes |

Single experiments run under their own names, for example `actapint figure_4_20`.
Within `section4` the cost is concentrated in five figures: Figure 4.17 alone takes
10 minutes, Figure 4.22 two and a half, Figure 4.13 two and a third, and Figures
4.5 and 4.11 about one and a half each; the other fifteen together take under a
minute. Every nonlinear experiment carries a Newton solve inside a time step
inside an iteration inside a parameter sweep, which is what that cost is.

The runs are deterministic. Random starting iterates come from MATLAB's
`rng('default')` stream, reproduced through NumPy's legacy `RandomState(5489)`
with MATLAB's column-major fill, so repeated runs agree bit for bit up to the
point where a curve starts amplifying roundoff — the divergent tails of Figures
3.8, 3.13, 4.5(b) and 4.9, which vary by tens of percent from run to run without
changing which entries diverge.

That is measured rather than argued. Rerunning `section3` and `section4` from
scratch and comparing every number of the committed metrics files entry by entry
— 5555 of them in Chapter 3 and 7202 in Chapter 4 — leaves 75 and 54 that move
by more than a part in a million, and every one of them is a quantity that is
either a difference of two iterates that have already converged or a ratio taken
at the roundoff floor. In Chapter 4 all 54 are values of one against one and a
half times machine epsilon in the tails of Figure 4.10. In Chapter 3 they are the
increments of Figure 3.16 at `nu = 1e-6` and `alpha = 1e-3`, the case that
section describes as stalling on the `1/alpha` amplification of the
diagonalization; the last GMRES residual of two panels of Figure 3.15, at `1e-14`;
the Newton residual of the reference of Figure 3.14, between `7e-14` and `6e-13`;
and the final values of runs in that figure's weight scan that diverge past
`1e10` and are reported as divergent. No number quoted in these reports is among
them.

The test suite checks the load-bearing pieces of all three chapters in about 70
seconds:

```bash
python -m pytest tests -q
```

## Parameters recovered from the published figures

Every experiment below needed a value that neither the article nor the committed
scripts state. Each was recovered by matching the published figure, and each is
recorded in the corresponding experiment's `parameters` block in
[`data`](data) so the choice is visible in the machine-readable output.

| Item | Quantity | Recovered value | How |
|---|---|---|---|
| Figure 3.1 | the sampled diffusion parameters | `nu = 2^0, ..., 2^-7` | tick calibration of the published axes |
| Figure 3.1 | the domain length `L`, declared global and never assigned | `8.2` | the length the script builds `x` with |
| Figures 3.9, 3.10 | the advection coefficient of the ADE panels | `1/2` | the only value that puts the curves on the published ones |
| Figures 3.10, 3.11(b), Table 3.1 | the mesh ratio `rho_num` | one ratio shared by all step counts | with the per-`N_t` minimizer neither the error growth nor the condition numbers appear |
| Figure 3.12 | the BVM matrix | (3.39b), centred in time closed by one backward Euler step | stated in the text, no script |
| Figure 3.14 | what the panels plot | `|U^k - U*|_inf` is the current comparison convention | shared first markers support an initial-error interpretation but do not identify its norm; the independent audit also records spatial L2, scaled L2, RMS and increments |
| Figure 3.14 | the starting iterate of the `nu = 0.02` panel | uniform random on `(-1,1)` is the closest tested start | this choice gives a similar initial max-norm error and averaged trajectory; the source does not specify the random stream |
| Figure 3.14 | the reduced model behind the `nu = 0.02` panel | source configuration still needed | the tested reduced models and finite weight optimizations do not reach the published contraction factor |
| Figure 4.2 | the coarsening factor `J` | `4`, against the `10` in the text | at `J = 10` the curve sits a factor three low at every iteration |
| Figure 4.2 | the starting iterate | the sequential coarse solution | the script's committed alternative starts elsewhere |
| Figure 4.5(a) | the initial condition | `sin(2 pi x)^2`, against `sin(2 pi x)` in the text | the script's value, and the only one that reproduces the panel |
| Figure 4.6 | the starting iterate | uniform random on `(0,1)` | a zero iterate starts the curves a factor eight low |
| Figure 4.10 | the initial condition and the starting iterate | `sin(2 pi (1-x))^2` from `u0` at every level for the first three panels, the script's `sin(2 pi (1-x)^2)^2` from the random draw for the divergent one | both are lines of the script, and each reproduces its own panels at every marker while missing the others by factors of 4.6 to 17 |
| Figure 4.11 | the three viscosities | `0.5`, `1/64`, `0.006` | panel-by-panel matching; stated nowhere |
| Figure 4.12 | the spatial grid | 128 points on `(0,1)`, not 257 on `[-1,1]` | only the former gives the printed factors |
| Figure 4.12 | the fine solver | SDIRK23 as in the script; the text says SDIRK22 | indistinguishable, the two agree to three digits |
| Figure 4.14 | the coarsening factor `J` | `20`, against the `10` in the text | at `J = 10` the curves bend and sit four times the published factor; the deviation over the six curves is 0.05 decades at 20, 0.24 at 19, 0.60 at 21, 3.47 at 10 |
| Figures 4.14–4.16 | the starting iterate | the random draw for heat, the script's committed overwrite by `u0` for the wave equation | the six heat curves start at six different values, which only a draw gives; the three wave curves start at the value `u0` in every window gives |
| Figure 4.17 | the three viscosities | `1`, `0.01`, `0.0001` | read off the published legend |
| Figures 4.20, 4.22 | the unlabelled dash-dot reference line | one time step | the line the published panels draw is undefined in the text |
| Figure 4.22 | smoothing steps and damping | two steps from the caption, `eta = 0.28` from the script | the script's four steps give a visibly faster curve |

## Corrections to the article and its scripts

### Scripts that cannot run as committed

* `MGRiT_Heat_ADE.m` contains `nu=0.002max;`, which is not valid MATLAB. The port
  uses `nu=0.002`, matching the branch, title and sweep around it.
* `Solution_Wave.m` sets `A2(1,1)=nu/dx^2` in its Neumann branch, but `nu` is never
  assigned in that script. The port uses `c^2/dx^2`, the coefficient of the
  surrounding branches, and panel 2.4(b) then matches at correlation 1.000.
* `SWR_for_ADE.m` declares `global L` and never assigns it, so `u0` cannot
  evaluate. The port uses `L = 8.2`.

### Where the script and the published figure disagree

Resolved in favour of the figure, since the figure is the published evidence:

* Equation (2.6) and `Solution_Bergers.m` both specify the flux `0.5 (u^2)_x`; the
  panels of Figure 2.3 are only reproduced with `(u^2)_x`, and so are Figures
  4.5(b) and 4.11. Figures 4.13 and 4.22 are reproduced with the committed factor,
  so the two coefficients are both kept, named, and stated per experiment.
* `SWR_for_ADE.m` scales a centred first difference by `dt/dx` rather than
  `dt/(2 dx)`, transporting at speed two; Figure 3.1(b) needs the consistent
  coefficient.
* `Iterative_ParaDiag_ADE.m` sets `theta = 1`, backward Euler, where the text and
  the caption of Figure 3.16 say trapezoidal rule.
* `DiagG_Parareal_Heat_Wave.m` omits the factor `1 - alpha` that equation (4.28)
  puts in front of the initial condition of the coupled window system. The wave
  panels are reproduced as committed, the heat panels only with the factor.
* `STMG_Bergers.m` commits four smoothing steps and overwrites its damping
  parameter with `0.28` inside the loop; Figure 4.22 needs the two steps of its
  caption together with that overwritten damping.
* `MGRiT_Heat_ADE.m` cannot draw all four panels of Figure 4.10 in one state. The
  first three need the plain `sin(2 pi (1-x))^2` started from `u0` at every level,
  which is the commented-out `U_Ini(:,n+1)=u0`; the divergent panel needs the
  committed chirp started from the committed random draw. Each setup reproduces
  its own panels at every marker and misses the others by factors of 4.6 to 17.
* `Direct_ParaDiag_ADE.m` and `Direct_ParaDiag_Wave.m` apply `It/V` as a formed
  matrix, and `Iterative_ParaDiag_RoundError.m` solves the blocks of `P_alpha`
  that the port had inverted. Both choices decide how much roundoff the run
  realizes, by decades in Figures 3.10 to 3.12 and by one in Figure 3.18.

### Printed statements that the reproduction contradicts

* Equation (3.2c) states an inequality that its own `R0` violates at `y = y_min`;
  what Figure 3.1(a) plots is the right-hand side.
* Theorem 3.1's `p* = p_tilde* s / nu` does not reproduce Figure 3.1(b); the
  script's `p_tilde* nu / l` does.
* The Parareal factor printed in Figure 4.9 for the ADE at `nu = 0.01` is `0.9986`,
  just below one; it is `1.0501`, just above. The same routine reproduces the other
  five maxima of that figure and both divergent factors quoted for `nu = 0.002` to
  four digits, which is why this is read as a typo. Either way Parareal does not
  converge there, which is the point the chapter makes.
* The legend of Figure 4.5(b) has its `nu = 1` and `nu = 0.02` entries swapped: the
  curve labelled `nu = 1` is the diverging one.
* `Direct_ParaDiag_ADE.m` commits `adc = 0`, the heat equation, so the advection
  coefficient behind the published ADE panels of Figures 3.9 and 3.10 is absent
  from the repository; it is `1/2`, not the unit speed of equation (2.5).

### Quirks reproduced deliberately

These are not corrected, because the published figures depend on them:

* The periodic grids of Chapter 2 carry both `x = 0` and `x = 1` as unknowns, so
  the effective period is `1 + dx`.
* `Solution_Heat_ADE.m` runs a trapezoidal loop in its periodic branch and then
  overwrites the result with an IMEX loop; the port uses the IMEX loop, which is
  what the script produces.
* `STMG_Heat_ADE.m` forms the residual that drives the coarse grid correction from
  the iterate before the last smoothing step, because it reuses the loop variable
  of its Jacobi sweep.
* `STMG_Bergers.m` freezes the previous coarse level at the restricted iterate, so
  its coarse problem decouples into one nonlinear equation per level instead of
  being a sequential solve, and its first coarse level carries no initial
  condition.
* The Parareal and MGRiT scripts return the last Newton iterate when the cap is hit
  instead of failing, which is what lets the divergent panels of Figures 4.5(b) and
  4.11 run to the end.
* Parareal's coarse values are updated in place inside the sequential sweep, so `G`
  is applied to the new iterate while the subtracted coarse term still belongs to
  the previous one.

## What is not reproduced

* Table 4.1's historical weak/strong scaling measurements on up to 262,144 cores.
  The new `stmg_fem_3d` experiment reconstructs its P1/dG(3) discretization with
  49,152 tetrahedra and 7,471 spatial degrees of freedom. At 2, 4 and 8 time steps
  it needs seven cycles and agrees with a direct sequential solve to relative
  error below `5e-9`. The [2026-11-11 campaign](REPRODUCTION_CHECKS_2026-11-11.md)
  adds MPI time-slab distribution, coarse rank agglomeration and measured scaling
  on current hardware; it does not recreate the historical BlueGene/Q machine.
* Figure 3.14 at `nu = 0.02`: the specified coarse-model weights produce only a
  small acceleration. Independent residual-form and global sparse updates agree
  with the existing diagonalized implementation to about `1e-14`; the 24-case
  audit records alternative starts and error norms. The earlier finite scans
  do not establish impossibility for all diagonal weights. The original run
  configuration is needed to identify the difference. See the dated report.

Differences remain open inside experiments that are otherwise reproduced, each
bounded and discussed in its section report:

* the constant by which the two rough-source panels of Figures 3.5 and 3.6 are
  drawn, 2.447 and 1.469 times the maximum of the reference solution they plot.
  It is a normalization and not a model: it is constant to better than a per cent
  over every window that can be read, the two smooth panels need no such factor,
  and fitting `(nu, sigma)` to the published curve with no free scale lands exactly
  on the committed parameters. What it cannot be is the denominator the article
  prints for `err^k_n`, since that would make the initial curve touch one.
* the residual 0 to 3.5 decades of roundoff that the published Figure 3.11(a)
  still carries over this reproduction at `N_t = 10` to `25`, down from 0.8 to 10
  decades before the scripts' `It/V` arithmetic was reproduced.
* the last two or three markers of the heat and `nu = 0.1` panels of Figure 4.10,
  where the published curves flatten towards `1e-14` and the reproduced ones keep
  falling, four decades under the truncation level the figure draws.
