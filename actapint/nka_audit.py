"""Independent residual-form checks of the unresolved Figure 3.14 curves.

The implementation solves (B + Phi tensor Jbar) delta = b - B U - f(U).
The existing generator solves for the next iterate instead. A sparse global
solve is available as an independent reference for the diagonalized solve.
Several observables are saved; none is chosen by fitting the published curve.
"""

from __future__ import annotations
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
from .section3 import paradiag_bvm as p


def correction(
    b,
    iterate,
    time,
    advection,
    diffusion,
    weights,
    *,
    method="diagonalized",
    spectrum=None,
):
    n = iterate.shape[1]
    jac = diffusion + 2 * advection @ sparse.diags(iterate.mean(axis=0))
    nonlinear = (advection @ iterate.T**2 + diffusion @ iterate.T).T
    residual = b - time @ iterate - nonlinear
    if method == "global":
        matrix = sparse.kron(sparse.csc_matrix(time), sparse.eye(n)) + sparse.kron(
            sparse.diags(weights), jac
        )
        return spsolve(matrix.tocsc(), residual.ravel()).reshape(iterate.shape)
    if method != "diagonalized":
        raise ValueError(method)
    if spectrum is None:
        values, v = np.linalg.eig(time / weights[:, None])
        inverse = np.linalg.inv(v)
    else:
        values, v, inverse = spectrum
    rhs = inverse @ (residual / weights[:, None])
    solved = np.array(
        [
            splu((jac + lam * sparse.eye(n)).astype(complex).tocsc()).solve(
                row.astype(complex)
            )
            for lam, row in zip(values, rhs)
        ]
    )
    return (v @ solved).real


def observables(values, dx):
    spatial_l2 = np.linalg.norm(values, axis=1)
    return dict(
        linf=float(np.abs(values).max()),
        max_spatial_l2=float(spatial_l2.max()),
        max_spatial_l2_scaled=float(np.sqrt(dx) * spatial_l2.max()),
        rms=float(np.sqrt(np.mean(values**2))),
    )


def trajectory(nu, horizon, dx, dt, start, variant, maxiter=100):
    if start not in ("initial", "symmetric", "random") or variant not in (
        "average",
        "nka",
    ):
        raise ValueError("unknown initial state or quasi-Newton variant")
    a, d, x = p.burgers_operators(nu, dx)
    a = sparse.csc_matrix(a)
    d = sparse.csc_matrix(d)
    u0 = p.initial_condition(x)
    steps = int(round(horizon / dt))
    dt = horizon / steps
    ref, res = p.all_at_once_solution(a.toarray(), d.toarray(), u0, steps, dt)
    if res > 1e-9:
        raise ArithmeticError(f"reference residual {res}")
    time = p.bvm_matrix(steps, dt)
    b = np.zeros((steps, len(x)))
    b[0] = u0 / (2 * dt)
    weights = np.ones(steps) if variant == "average" else p.nka_weights(nu, dt, steps)
    values, v = np.linalg.eig(time / weights[:, None])
    spectrum = values, v, np.linalg.inv(v)
    if start == "initial":
        u = np.tile(u0, (steps, 1))
    else:
        u = np.random.default_rng(p.RANDOM_SEED).uniform(
            -1 if start == "symmetric" else 0, 1, (steps, len(x))
        )
    records = [dict(iteration=0, error=observables(u - ref, dx))]
    termination = "iteration_limit"
    for k in range(maxiter):
        change = correction(b, u, time, a, d, weights, spectrum=spectrum)
        u += change
        residual = b - time @ u - (a @ u.T**2 + d @ u.T).T
        records.append(
            dict(
                iteration=k + 1,
                error=observables(u - ref, dx),
                increment=observables(change, dx),
                residual_linf=float(np.abs(residual).max()),
            )
        )
        if not np.isfinite(u).all():
            raise ArithmeticError("nonfinite quasi-Newton iterate")
        if records[-1]["error"]["linf"] < 1e-6:
            termination = "error_tolerance"
            break
        if records[-1]["increment"]["linf"] > 1e5:
            termination = "divergence_limit"
            break
    return dict(
        nu=nu,
        horizon=horizon,
        dx=dx,
        dt=dt,
        start=start,
        variant=variant,
        seed=p.RANDOM_SEED,
        termination=termination,
        error_tolerance=1e-6,
        max_iterations=maxiter,
        reference_residual=res,
        weight_range=[float(weights.min()), float(weights.max())],
        records=records,
    )


def run_nka_audit(output_dir, quick=False):
    runs = []
    for nu in p.NKA_NU:
        for horizon in p.NKA_HORIZONS:
            for start in ("symmetric", "initial", "random"):
                for variant in ("average", "nka"):
                    print(
                        f"  NKA audit nu={nu} T={horizon} {start} {variant}", flush=True
                    )
                    runs.append(
                        trajectory(
                            nu,
                            horizon,
                            1 / 50 if quick else p.NKA_DX,
                            p.NKA_DT,
                            start,
                            variant,
                            30 if quick else 100,
                        )
                    )
    payload = dict(
        status="independent implementation and observables; Figure 3.14 matching remains an acceptance check",
        method="published full-Jacobian Frobenius weights from coarse trapezoidal trajectory; residual-form quasi-Newton",
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        runs=runs,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "figure_3_14_audit.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return str(path)
