# GPU acceleration study

## Decision from profiling

Function-level profiling was performed before selecting a GPU target. In the
quick paper suite, Figure 4.5 required 43.06 of 62.95 seconds. Batched Burgers
fine propagation accounted for 38.11 seconds, whereas all FFT calls required
0.007 seconds and all GMRES calls required 0.251 seconds.

The first CUDA backend therefore targets the operation with both the largest
measured cost and genuine time-slice parallelism: every Parareal iteration has
many independent nonlinear fine propagations. It does not move small FFTs to
the GPU merely because a GPU is present.

## Implementation

`CupyBurgersFineStepper` keeps both spatial operators on the device and advances
all time slices as an `(Nx, Nt)` batch. Each SDIRK22 stage forms a batch of
independent Newton Jacobians and solves them through CuPy's batched dense
linear algebra. Only one input and one output transfer are needed for a whole
fine-propagator batch.

The coarse Parareal correction remains causal and executes on the CPU. Other
experiments currently retain their SciPy implementations, so this is a hybrid
CPU/GPU backend rather than a claim that every line runs on CUDA.

## Tesla T4 result

The benchmark used a Tesla T4, CUDA 12.2, CuPy 13.6.0, double precision, one CPU
BLAS thread, and the paper-scale Burgers batch:

| Metric | Result |
|---|---:|
| spatial points | 128 |
| independent time slices | 40 |
| SDIRK22 substeps | 32 |
| CPU batch | 2.893 s |
| GPU batch | 0.246 s |
| kernel speedup | 11.76x |
| CPU/GPU maximum absolute difference | $2.33\times 10^{-15}$ |

The complete paper-validation wall time falls from 263.57 seconds on the CPU to
67.92 seconds with the hybrid GPU backend, a 3.88x end-to-end speedup. The
reported Figure 4.5 stopping iterations remain identical:

- ADE: 14, 24, and 35 iterations;
- Burgers: 14, 21, and 25 iterations.

Late iterations below the requested $10^{-10}$ stopping scale can differ near
roundoff because CPU SuperLU and GPU batched LU use different floating-point
operation orders. Those differences do not change any reported stopping
iteration or qualitative conclusion.

The machine-readable benchmark is
[`docs/data/gpu_benchmark_t4.json`](data/gpu_benchmark_t4.json). Reproduce the
kernel measurement with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_gpu.py
```

## Remaining opportunities

1. Replace dense batched Newton solves with a cyclic tridiagonal CUDA kernel.
   The present approach is effective at $N_x=128$ but scales as
   $\mathcal{O}(N_x^3)$ per Newton solve instead of exploiting the banded
   operator.
2. Keep the complete nonlinear Parareal state on the device and implement the
   coarse scan with a parallel prefix formulation, removing per-iteration
   host/device transfers and the serial correction bottleneck.
3. For larger ParaDiag problems, use batched shifted banded solvers after the
   time FFT. At the current $100\times100$ paper grid, profiling shows that the
   FFT and GMRES paths are too small to justify a separate GPU implementation.
4. Distribute time-slice batches across multiple GPUs and overlap fine
   propagation, communication, and coarse correction.
5. Add strong- and weak-scaling experiments. The current numbers establish
   single-GPU kernel and end-to-end acceleration, not multi-GPU scalability.
