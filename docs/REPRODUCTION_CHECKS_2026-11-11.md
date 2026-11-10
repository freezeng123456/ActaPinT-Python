# NKA reconstruction and distributed STMG, 2026-11-11

Figure 3.14 now has a reconstruction with explicit experimental choices, a
closed-form verification of the full-Jacobian NKA weights, and the original
EPS plot coordinates. The key control is **the same starting vector for both
methods**: a coarse-trajectory start improves both methods and cannot itself
be counted as an NKA improvement.

The Table 4.1 solver now distributes time slabs over MPI ranks, exchanges
neighbouring time boundaries, and agglomerates rank pairs on coarse time grids.
Every solver iteration keeps its vectors distributed. Small-system comparisons
check it against a separately assembled global matrix and the serial V-cycle.

## Figure 3.14: reconstruction from the equations

The [Acta paper](https://doi.org/10.1017/S0962492924000072) supplies viscosities
`0.02, 0.002`, horizons `1.3, 0.7`, fine spatial step `1/200`, and coarse spatial
step `1/20`. We use 200 time steps, the count stated for the preceding Figure 3.13
and used in the public neighbouring `Direct_ParaDiag_Burgers.m` example.
The physical initial condition is `sin(2*pi*x)` and the source term is zero,
as in that script. This is an explicit reconstruction choice:
Figure 3.14's paragraph does not specify its time step, starting vector or
plotted error definition.

The canonical grid has 200 **distinct** periodic nodes. A second complete run
uses 201 nodes including both endpoints, following the neighbouring MATLAB
script's convention. Both use centred conservative Burgers differences,
the paper's centred-time BVM with backward-Euler closure, and coarse trapezoidal
time integration to freeze the NKA weights before the fine-grid iteration.

For the semidiscrete problem `u' = -f(u)`, the correction is

```text
(B ⊗ I + diag(phi) ⊗ J(mean(U))) delta = b - B U - f(U)
U_next = U + delta
```

There are three independently cross-checked linear solves: a global sparse
solve, diagonalization of `diag(phi)^-1 B`, and the earlier next-iterate
formulation using `B^-1 diag(phi)`. The new tests additionally verify the
Burgers Jacobian by a directional finite difference and check discrete mass
conservation on the periodic grid.

For centred periodic differences, write `J(u) = D + N(u)`, where `D` is the
diffusion matrix and `N(u) = D_center diag(u)`. Direct calculation gives

```text
<D, N(u)>_F = 0
||D||_F^2 = 6 n nu^2 / h^4
<N(u), N(v)>_F = dot(u, v) / (2 h^2)

phi_j = (12 n nu^2 / h^2 + dot(u_j, mean(u)))
        / (12 n nu^2 / h^2 + dot(mean(u), mean(u)))
```

`actapint/nka_reproduction.py` implements this identity without constructing
all coarse Jacobian matrices. Tests compare it to explicit matrix traces and
the Frobenius normal equations. It includes the diffusion term, which dominates
the weights at `nu=0.02`; omitting that term would change the algorithm.

On the canonical coarse grid, the weight ranges are:

| nu | T | min(phi) | max(phi) |
|---:|---:|---:|---:|
| 0.02 | 1.3 | 0.966752 | 1.054118 |
| 0.02 | 0.7 | 0.956495 | 1.043681 |
| 0.002 | 1.3 | 0.487830 | 1.582450 |
| 0.002 | 0.7 | 0.619543 | 1.235339 |

### What explains the apparent improvement from a coarse initial guess

Each configuration is run with the public script's uniform `[0,1]` random
vector, a symmetric `[-1,1]` random vector (both MT19937, seed 5489), the initial
sine wave repeated in time, and the interpolated coarse trapezoidal trajectory. Each starting vector is identical for averaging and NKA. Stopping
uses the true maximum error `<= 1e-6` against a converged all-at-once Newton
reference, with a separate divergence threshold and a 100-iteration cap.

| nu | T | Start | Averaging iterations | NKA iterations |
|---:|---:|---|---:|---:|
| 0.02 | 1.3 | uniform [0,1] | 14 | 14 |
| 0.02 | 1.3 | symmetric [-1,1] | 16 | 15 |
| 0.02 | 1.3 | repeated initial condition | 12 | 12 |
| 0.02 | 1.3 | coarse TR | 10 | 10 |
| 0.02 | 0.7 | uniform [0,1] | 11 | 11 |
| 0.02 | 0.7 | symmetric [-1,1] | 10 | 10 |
| 0.02 | 0.7 | repeated initial condition | 9 | 9 |
| 0.02 | 0.7 | coarse TR | 7 | 7 |
| 0.002 | 1.3 | uniform [0,1] | diverged | diverged |
| 0.002 | 1.3 | symmetric [-1,1] | diverged | diverged |
| 0.002 | 1.3 | repeated initial condition | diverged | 71 |
| 0.002 | 1.3 | coarse TR | 50 | 64 |
| 0.002 | 0.7 | uniform [0,1] | diverged | diverged |
| 0.002 | 0.7 | symmetric [-1,1] | 38 | 42 |
| 0.002 | 0.7 | repeated initial condition | 21 | 19 |
| 0.002 | 0.7 | coarse TR | 27 | 23 |

The warm-start experiment explains a **confounder in our curve comparisons**:
matching the paper's later NKA points with a better starting guess does not
establish that NKA caused the acceleration. It does not establish which initial
state the authors used. The endpoint-duplicated control also fails to recover
all four published curves under one shared protocol.

![Original EPS points and recomputed increments](figures/nka_reconstruction_2026-11-11.svg)

The overlay uses maximum update increments, indexed from zero as in the public
neighbouring MATLAB script. Raw results also retain true solution errors,
spatial L2 errors, scaled L2 errors, RMS errors and nonlinear residuals. These
observables are kept separate. The paper's red curve is still an external
comparison, not an acceptance target achieved by parameter fitting.

### Original-source evidence

The [arXiv source archive](https://arxiv.org/src/2503.13526) contains the two
original MATLAB-exported EPS files. `scripts/extract_nka_eps.py` reads their
polylines and calibrates them using the labelled axes. This replaces screenshot
point estimates with numeric vector coordinates; the EPS rounds coordinates to
0.001 PostScript units. Both methods have **exactly identical initial marker
coordinates** for each horizon. The archive also names the generating script
`Sec3_6_1_ParaDiag1_Burgers_NKA.m` in a commented TeX margin note, but does not
include that script. Its sole `.fig` file belongs to PFASST, not NKA.

Archive SHA256:
`cbd4ded34471623ce546eff14d834ec92b7d6088ab94a644a165854e52179672`.
The original EPS hashes and calibration constants are in
`docs/data/reproduction-2026-11-11/figure_3_14_eps.json`.

The reconstruction is consequently a usable mathematical baseline. The specific
large acceleration in Figure 3.14 is not recovered by the published
full-Jacobian weights under these controlled settings. Future algorithm changes
should be compared against these shared-start runs and include coarse-model
setup cost, rather than using a different initialization only for NKA.

## Distributed Table 4.1 implementation

`actapint/stmg_mpi.py` preserves the 49,152-tetrahedron P1 spatial discretization,
7,471 interior vertices, dG(3) time discretization, time step 0.1, V(1,1) cycle,
time damping 1 and spatial damped Gauss-Seidel parameter 1.285.

Each rank owns a contiguous group of time slabs and a full spatial mesh. Matrix
application exchanges only the final slab with its right neighbour. Restriction
is local while a rank owns at least two slabs. Once it owns one slab, odd ranks
send to their even neighbours; those neighbours form the next communicator.
Prolongation returns corrections to the odd ranks. Only one rank reaches the
single-slab direct coarse solve. There is no full-vector gather in the solver
or benchmark. Direct LU factors are created only for blocks that use them.

The benchmark fixes each slab's random seed by its global index, so strong
scaling uses exactly the same problem at every rank count. It measures the
maximum elapsed time across ranks after a barrier, includes communication and
convergence reductions, excludes setup, performs one warm-up and three measured
solves, and saves every timing and residual history. A separate zero-RHS test
checks error reduction. Tests compare complete distributed V-cycles and final
solutions to both serial multigrid and an independent sparse global solve.

Weak scaling uses two slabs per rank, as in Table 4.1. Strong scaling uses 512
slabs, hence exactly **15,300,608 unknowns**, as in its smaller strong-scaling
block. MPI rank count, actual hostnames, CPU affinity, library versions and
source hashes are included in each result file. Cross-node and single-host
runs must be reported separately.

## Reproduction commands

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m pip install -e '.[test,mpi]'
# Use the cluster MPI module in production; mpi4py + mpich wheels also work locally.
python -m actapint.nka_reproduction --output results/nka_periodic.json
python -m actapint.nka_reproduction --duplicate-endpoint --output results/nka_duplicate_endpoint.json
mpiexec -n 4 python -m scripts.check_stmg_mpi
mpiexec -n 4 python -m scripts.benchmark_stmg_mpi --output results/weak-p4.json
mpiexec -n 4 python -m scripts.benchmark_stmg_mpi --steps 512 --output results/strong-p4.json
```

The MPI implementation follows the direct NumPy-buffer communication interface
in the [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html).

<!-- ACTUAL_SCALING_RESULTS -->
## First completed scaling measurements

The full local MPI matrix has completed on an Apple M3 Pro (12 physical cores,
36 GiB, one host). Each cell has one warm-up and three measured solves; all
measured and homogeneous checks take seven cycles. These are single-host
measurements, separate from the SCNet two-node campaign.

| Ranks | Weak median (s) | Strong 512-slab median (s) |
|---:|---:|---:|
| 1 | 0.545 | 95.540 |
| 2 | 0.551 | 58.898 |
| 4 | 0.919 | 45.525 |
| 8 | 1.159 | 39.953 |

On SCNet XiongHeng, job 25558592 passed the two-node 4-rank independent global
solve comparison (maximum solution difference `8.42e-12`). Job 25558675 has
completed the five weak-scaling cells below, all in seven cycles. P=1 uses one
host; P>=2 uses both Hygon C86 7185 nodes. The full 512-slab strong matrix and
the iterative forward-substitution comparison are still running under the
protocol in `run-plan.json`; their final measured results will be appended.

| Ranks | Weak median (s) | Actual hosts |
|---:|---:|---:|
| 1 | 1.479 | 1 |
| 2 | 1.892 | 2 |
| 4 | 2.315 | 2 |
| 8 | 2.775 | 2 |
| 16 | 3.162 | 2 |

Full local tests: **116 passed, one CUDA skip**. Python 3.10/3.12 and MPI
1/2/4-rank CI checks passed in
[run 42887337131](https://github.com/freezeng123456/ActaPinT-Python/actions/runs/42887337131).
CI-only `UCX_TLS=tcp,self,sm` avoids an unusable hosted-runner RDMA adapter;
cluster jobs keep their native transport configuration. The setting follows
the [UCX documentation](https://openucx.readthedocs.io/en/master/faq.html).
