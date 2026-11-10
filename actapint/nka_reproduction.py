"""Explicit, independently checkable Figure 3.14 reconstruction.

Default grid has 200 distinct periodic nodes. The endpoint-duplicated grid used
in the public neighbouring MATLAB example is available as a separate control.
The paper supplies spatial steps, viscosities and horizons; time step, seed,
initial iterate and plotted observable are explicit reconstruction choices.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .operators import periodic_first_derivative, periodic_negative_laplacian
from .nka_audit import correction, observables
from .section3.paradiag_bvm import bvm_matrix, trapezoidal_steps, all_at_once_solution


def operators(nu, dx, duplicate_endpoint=False):
    n = round(1 / dx) + int(duplicate_endpoint)
    x = np.arange(n) * dx
    return (
        0.5 * periodic_first_derivative(n, dx).toarray(),
        periodic_negative_laplacian(n, dx, nu).toarray(),
        x,
    )


def frobenius_weights(states, nu, dx):
    """Closed form of full-Jacobian Frobenius NKA, including diffusion.

    For J(u)=nu/h^2*tridiag(-1,2,-1)+D_center*diag(u),
    <D,N(u)>=0 and <N(u),N(v)>=dot(u,v)/(2*h^2).
    Multiplying numerator and denominator by 2*h^2 yields this formula.
    """
    mean = states.mean(axis=0)
    diffusion_norm = 12 * states.shape[1] * nu**2 / dx**2
    denom = diffusion_norm + mean @ mean
    if denom <= 0:
        raise ValueError("zero averaged Jacobian Frobenius norm")
    return (diffusion_norm + states @ mean) / denom


def problem(nu, horizon, steps=200, duplicate_endpoint=False):
    dx, coarse_dx = 1 / 200, 1 / 20
    dt = horizon / steps
    a, d, x = operators(nu, dx, duplicate_endpoint)
    ca, cd, cx = operators(nu, coarse_dx, duplicate_endpoint)
    coarse, _ = trapezoidal_steps(ca, cd, np.sin(2 * np.pi * cx), steps, dt)
    weights = frobenius_weights(coarse[1:], nu, coarse_dx)
    interp_x, interp_u = cx, coarse[1:]
    if not duplicate_endpoint:
        interp_x = np.r_[cx, 1.0]
        interp_u = np.c_[coarse[1:], coarse[1:, 0]]
    warm = np.array([np.interp(x, interp_x, row) for row in interp_u])
    u0 = np.sin(2 * np.pi * x)
    reference, residual = all_at_once_solution(a, d, u0, steps, dt)
    if residual > 1e-9:
        raise ArithmeticError(f"Newton reference residual {residual}")
    time = bvm_matrix(steps, dt)
    b = np.zeros((steps, len(x)))
    b[0] = u0 / (2 * dt)
    return a, d, x, time, b, weights, warm, reference, residual


def run_case(data, nu, horizon, start, method, maxiter=100):
    a, d, x, time, b, weights, warm, reference, reference_residual = data
    steps = len(time)
    if start == "uniform":
        initial = np.random.RandomState(5489).random_sample(b.shape)
    elif start == "symmetric":
        initial = 2 * np.random.RandomState(5489).random_sample(b.shape) - 1
    elif start == "initial":
        initial = np.tile(np.sin(2 * np.pi * x), (steps, 1))
    elif start == "coarse_tr":
        initial = warm.copy()
    else:
        raise ValueError(start)
    w = weights if method == "nka" else np.ones(steps)
    eig, v = np.linalg.eig(time / w[:, None])
    spectrum = eig, v, np.linalg.inv(v)
    u = initial.copy()
    a, d = sparse.csc_matrix(a), sparse.csc_matrix(d)
    records = [dict(iteration=0, error=observables(u - reference, 1 / 200))]
    reason = "iteration_limit"
    for k in range(maxiter):
        delta = correction(b, u, time, a, d, w, spectrum=spectrum)
        u += delta
        residual = b - time @ u - (a @ u.T**2 + d @ u.T).T
        records.append(
            dict(
                iteration=k + 1,
                error=observables(u - reference, 1 / 200),
                increment=observables(delta, 1 / 200),
                residual_linf=float(abs(residual).max()),
            )
        )
        if records[-1]["error"]["linf"] <= 1e-6:
            reason = "solution_error_tolerance"
            break
        if records[-1]["increment"]["linf"] > 1e5:
            reason = "divergence_limit"
            break
    return dict(
        nu=nu,
        horizon=horizon,
        steps=steps,
        dt=horizon / steps,
        nx=len(x),
        start=start,
        method=method,
        seed=5489,
        rng="MT19937",
        initial_linf=float(abs(initial).max()),
        weights_min=float(w.min()),
        weights_max=float(w.max()),
        reference_residual=reference_residual,
        termination=reason,
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--duplicate-endpoint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for nu in (0.02, 0.002):
        for horizon in (1.3, 0.7):
            data = problem(nu, horizon, args.steps, args.duplicate_endpoint)
            for start in ("uniform", "symmetric", "initial", "coarse_tr"):
                for method in ("average", "nka"):
                    r = run_case(data, nu, horizon, start, method)
                    runs.append(r)
                    print(
                        nu,
                        horizon,
                        start,
                        method,
                        len(r["records"]) - 1,
                        r["termination"],
                        r["weights_min"],
                        r["weights_max"],
                        flush=True,
                    )
    out = dict(
        spatial_grid=(
            "duplicate_endpoint"
            if args.duplicate_endpoint
            else "distinct_periodic_nodes"
        ),
        initial_condition="sin(2*pi*x)",
        source_term="zero",
        fine_dx=1 / 200,
        coarse_dx=1 / 20,
        time_steps=args.steps,
        tolerance=1e-6,
        maxiter=100,
        stop="true max-norm error against converged all-at-once Newton reference",
        plot_indices="error at k=0 is the initial error; increment at index k-1 is update k",
        runs=runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
