from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu

from .integrators import (
    burgers_fine_step,
    burgers_ie_step,
    linear_step_matrix,
    repeated_linear_step,
)
from .operators import (
    burgers_operators,
    dirichlet_negative_laplacian,
    linear_ade_operator,
)
from .plotting import COLORS, save_figure, style_axis


def _prepare(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_metrics(output_dir, name, metrics):
    path = Path(output_dir) / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _plot_solution(t, x, u, title, path):
    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    image = ax.imshow(
        u,
        extent=(t[0], t[-1], x[0], x[-1]),
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set(xlabel="t", ylabel="x", title=title)
    style_axis(ax, grid=False)
    colorbar = fig.colorbar(image, ax=ax, label=r"$u(x,t)$", pad=0.025, aspect=30)
    colorbar.outline.set_edgecolor(COLORS["grid"])
    save_figure(fig, path)


def _plot_errors(series, title, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    markers = ("o", "s", "^", "D")
    for marker, (label, values) in zip(markers, series.items()):
        ax.semilogy(
            np.arange(len(values)),
            values,
            marker=marker,
            markevery=max(1, len(values) // 10),
            label=label,
        )
    ax.set(xlabel="Iteration", ylabel=r"Maximum error $\|e^k\|_\infty$", title=title)
    style_axis(ax)
    ax.legend()
    save_figure(fig, path)


def run_linear_solution(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    # Stability of the original IMEX-centered scheme requires nu*dt/dx^2=0.5.
    # Keep the original grid ratio in quick mode and shorten only the time span.
    dt = 0.001
    t_final = 0.25 if quick else 3.0
    dx = dt
    x = np.arange(0.0, 1.0 + 0.5 * dx, dx)
    interior = x[1:-1]
    n = interior.size
    nu, advection = 5e-4, 1.0
    d2 = -dirichlet_negative_laplacian(n, dx, scale=nu)
    e = np.ones(n)
    d1 = sparse.diags((-e, e), (-1, 1), shape=(n, n), format="csc") * (advection / (2 * dx))
    # MATLAB uses u_t = nu*u_xx - a*u_x and IMEX Euler.
    implicit = splu((sparse.eye(n, format="csc") - dt * d2).tocsc())
    steps = int(round(t_final / dt))
    u = np.zeros((x.size, steps + 1))
    u[:, 0] = np.sin(8 * np.pi * (1 - x) ** 2) ** 2
    for k in range(steps):
        u[1:-1, k + 1] = implicit.solve(u[1:-1, k] - dt * (d1 @ u[1:-1, k]))
    t = np.linspace(0, t_final, steps + 1)
    _plot_solution(t, x, u, "Advection-diffusion solution", output_dir / "solution_heat_ade.png")
    return _save_metrics(output_dir, "solution_heat_ade", {
        "dt": dt, "dx": dx, "steps": steps,
        "min": float(u.min()), "max": float(u.max()),
        "final_linf": float(np.linalg.norm(u[:, -1], np.inf)),
    })


def run_burgers_solution(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    dt = 1 / (160 if quick else 400)
    t_final = 1.0 if quick else 3.0
    dx = dt
    x = np.arange(0.0, 1.0 + 0.5 * dx, dx)
    n = x.size - 2
    a1, a2 = burgers_operators(n, dx, 5e-4, periodic=False)
    steps = int(round(t_final / dt))
    u = np.zeros((x.size, steps + 1))
    u[:, 0] = np.sin(8 * np.pi * (1 - x) ** 2) ** 2
    newton_tol = max(dt**2, dx**2) / 10
    for k in range(steps):
        old = u[1:-1, k]
        rhs = old - 0.5 * dt * (a2 @ old + a1 @ (old**2))
        u[1:-1, k + 1] = burgers_ie_step(a1, a2, 0.5 * dt, rhs, newton_tol)
    t = np.linspace(0, t_final, steps + 1)
    _plot_solution(t, x, u, "Viscous Burgers solution", output_dir / "solution_burgers.png")
    return _save_metrics(output_dir, "solution_burgers", {
        "dt": dt, "dx": dx, "steps": steps,
        "min": float(u.min()), "max": float(u.max()),
        "final_linf": float(np.linalg.norm(u[:, -1], np.inf)),
    })


def run_wave_solution(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    dt = 1 / (160 if quick else 400)
    t_final = 1.0 if quick else 3.0
    dx = dt
    x = np.arange(0.0, 1.0 + 0.5 * dx, dx)
    n = x.size - 2
    c = 0.2
    lap = dirichlet_negative_laplacian(n, dx, scale=c)
    zero = sparse.csc_matrix((n, n))
    eye = sparse.eye(n, format="csc")
    # z=[u;v], z_t + [[0,-I],[L,0]] z = 0
    a = sparse.bmat([[zero, -eye], [lap, zero]], format="csc")
    solve = splu((sparse.eye(2 * n, format="csc") + 0.5 * dt * a).tocsc())
    rhs_matrix = sparse.eye(2 * n, format="csc") - 0.5 * dt * a
    steps = int(round(t_final / dt))
    u = np.zeros((x.size, steps + 1))
    u[:, 0] = np.sin(8 * np.pi * (1 - x) ** 2) ** 2
    z = np.r_[u[1:-1, 0], np.zeros(n)]
    for k in range(steps):
        z = solve.solve(rhs_matrix @ z)
        u[1:-1, k + 1] = z[:n]
    t = np.linspace(0, t_final, steps + 1)
    _plot_solution(t, x, u, "Wave equation solution", output_dir / "solution_wave.png")
    return _save_metrics(output_dir, "solution_wave", {
        "dt": dt, "dx": dx, "steps": steps,
        "min": float(u.min()), "max": float(u.max()),
        "final_linf": float(np.linalg.norm(u[:, -1], np.inf)),
    })


def run_parareal_linear(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    n = 64 if quick else 128
    nt = 20 if quick else 40
    substeps = 8 if quick else 32
    t_final, nu = 4.0, 0.02
    coarse_dt = t_final / nt
    fine_dt = coarse_dt / substeps
    a = linear_ade_operator(n, 1 / n, nu, periodic=True)
    fine = repeated_linear_step(a, fine_dt, substeps, "sdirk22")
    coarse = linear_step_matrix(a, coarse_dt, "ie")
    x = np.linspace(0, 1, n)
    u0 = np.sin(2 * np.pi * x) ** 2
    reference = np.zeros((n, nt + 1))
    reference[:, 0] = u0
    for j in range(nt):
        reference[:, j + 1] = fine @ reference[:, j]
    current = np.tile(u0[:, None], (1, nt + 1))
    coarse_old = current.copy()
    errors = [float(np.max(np.abs(current - reference)))]
    for _ in range(max(50, nt + 1)):
        fine_values = np.column_stack([fine @ current[:, j] for j in range(nt)])
        updated = current.copy()
        for j in range(nt):
            new_coarse = coarse @ updated[:, j]
            updated[:, j + 1] = new_coarse + fine_values[:, j] - coarse_old[:, j + 1]
            coarse_old[:, j + 1] = new_coarse
        current = updated
        errors.append(float(np.max(np.abs(current - reference))))
        if errors[-1] <= 2e-14:
            break
    _plot_errors({"Parareal": errors}, "Parareal for ADE", output_dir / "parareal_heat_ade.png")
    return _save_metrics(output_dir, "parareal_heat_ade", {
        "parameters": {"Nx": n, "Nt": nt, "J": substeps, "nu": nu},
        "errors": errors, "iterations": len(errors) - 1,
        "final_error": errors[-1],
    })


def run_parareal_burgers(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    n = 48 if quick else 128
    nt = 16 if quick else 40
    substeps = 4 if quick else 32
    t_final, nu = 2.0 if quick else 4.0, 1.0
    coarse_dt = t_final / nt
    fine_dt = coarse_dt / substeps
    a1, a2 = burgers_operators(n, 1 / n, nu, periodic=True)
    x = np.linspace(0, 1, n)
    u0 = np.sin(2 * np.pi * x) ** 2
    fine = lambda y: burgers_fine_step(a1, a2, y, fine_dt, substeps, "sdirk22", 1e-12)
    coarse = lambda y: burgers_ie_step(a1, a2, coarse_dt, y, 1e-12)
    reference = np.zeros((n, nt + 1))
    reference[:, 0] = u0
    for j in range(nt):
        reference[:, j + 1] = fine(reference[:, j])
    current = np.tile(u0[:, None], (1, nt + 1))
    coarse_old = current.copy()
    errors = [float(np.max(np.abs(current - reference)))]
    for _ in range(nt):
        fine_values = [fine(current[:, j]) for j in range(nt)]
        updated = current.copy()
        for j in range(nt):
            new_coarse = coarse(updated[:, j])
            updated[:, j + 1] = new_coarse + fine_values[j] - coarse_old[:, j + 1]
            coarse_old[:, j + 1] = new_coarse
        current = updated
        errors.append(float(np.max(np.abs(current - reference))))
        if errors[-1] <= 1e-11:
            break
    _plot_errors({"Parareal": errors}, "Parareal for Burgers", output_dir / "parareal_burgers.png")
    return _save_metrics(output_dir, "parareal_burgers", {
        "parameters": {"Nx": n, "Nt": nt, "J": substeps, "nu": nu},
        "errors": errors, "iterations": len(errors) - 1,
        "final_error": errors[-1],
    })


def run_mgrit_linear(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    n = 64 if quick else 160
    nt = 20 if quick else 40
    substeps = 8 if quick else 20
    t_final, nu = 2.5 if quick else 5.0, 0.002  # fixes MATLAB typo: 0.002max
    coarse_dt = t_final / nt
    fine_dt = coarse_dt / substeps
    a = linear_ade_operator(n, 1 / n, nu, periodic=True)
    fine = repeated_linear_step(a, fine_dt, substeps, "sdirk22")
    coarse = linear_step_matrix(a, coarse_dt, "ie")
    x = np.linspace(0, 1, n)
    u0 = np.sin(2 * np.pi * (1 - x) ** 2) ** 2
    ref = np.zeros((n, nt + 1))
    ref[:, 0] = u0
    for j in range(nt):
        ref[:, j + 1] = fine @ ref[:, j]
    rng = np.random.default_rng(0)
    initial = rng.random((n, nt + 1))
    initial[:, 0] = u0
    p = initial.copy()
    p_coarse = p.copy()
    parareal = [float(np.max(np.abs(p - ref)))]
    for _ in range(min(25, nt)):
        fvals = np.column_stack([fine @ p[:, j] for j in range(nt)])
        updated = p.copy()
        for j in range(nt):
            cg = coarse @ updated[:, j]
            updated[:, j + 1] = cg + fvals[:, j] - p_coarse[:, j + 1]
            p_coarse[:, j + 1] = cg
        p = updated
        parareal.append(float(np.max(np.abs(p - ref))))
    m = initial.copy()
    m[:, 1] = fine @ u0
    mgrit = [float(np.max(np.abs(m - ref)))]
    for _ in range(25):
        updated = m.copy()
        f2 = np.zeros_like(m)
        g = np.zeros_like(m)
        for j in range(1, nt - 1):
            mid = fine @ m[:, j - 1]
            f2[:, j + 1] = fine @ mid
            g[:, j + 1] = coarse @ mid
        for j in range(1, nt - 1):
            updated[:, j + 1] = coarse @ updated[:, j] + f2[:, j + 1] - g[:, j + 1]
        m = updated
        mgrit.append(float(np.max(np.abs(m[:, 2:nt] - ref[:, 2:nt]))))
    _plot_errors({"Parareal": parareal[::2], "MGRiT": mgrit}, "Parareal vs MGRiT", output_dir / "mgrit_heat_ade.png")
    return _save_metrics(output_dir, "mgrit_heat_ade", {
        "parameters": {"Nx": n, "Nt": nt, "J": substeps, "nu": nu},
        "parareal_errors": parareal, "mgrit_errors": mgrit,
        "parareal_final": parareal[-1], "mgrit_final": mgrit[-1],
    })


def run_iterative_paradiag_ade(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    nx = 50 if quick else 100
    nt = 40 if quick else 100
    t_final, nu, alpha = 2.0, 1e-6, 1.0
    dt = t_final / nt
    a = linear_ade_operator(nx, 1 / nx, nu, periodic=False).tocsc()
    eye = sparse.eye(nx, format="csc")
    r1, r2 = eye + dt * a, -eye
    x = np.linspace(0, 1, nx)
    u0 = np.sin(2 * np.pi * x)
    b = np.zeros(nx * nt)
    b[:nx] = -r2 @ u0

    def k_matvec(vector):
        u = vector.reshape((nx, nt), order="F")
        z = np.empty_like(u)
        z[:, 0] = r1 @ u[:, 0]
        for j in range(1, nt):
            z[:, j] = r1 @ u[:, j] + r2 @ u[:, j - 1]
        return z.reshape(-1, order="F")

    da = alpha ** (np.arange(nt) / nt)
    inv_da = 1.0 / da
    c = np.zeros(nt)
    c[1] = 1.0
    eig_c = np.fft.fft(da * c)
    solvers = [splu((r1 + eig_c[j] * r2).tocsc()) for j in range(nt)]

    def psolve(vector):
        u = vector.reshape((nx, nt), order="F")
        transformed = np.fft.fft((da[:, None] * u.T), axis=0).T
        solved = np.column_stack([solvers[j].solve(transformed[:, j]) for j in range(nt)])
        result = (inv_da[:, None] * np.fft.ifft(solved.T, axis=0)).T
        return np.real_if_close(result).reshape(-1, order="F")

    residuals = []
    operator = LinearOperator((nx * nt, nx * nt), matvec=k_matvec, dtype=float)
    preconditioner = LinearOperator((nx * nt, nx * nt), matvec=psolve, dtype=float)
    x0 = np.tile(0.125 * u0, nt)
    solution, info = gmres(
        operator, b, x0=x0, M=preconditioner, rtol=1e-12, atol=0.0,
        restart=min(100, nt), maxiter=1000,
        callback=lambda value: residuals.append(float(value)),
        callback_type="pr_norm",
    )
    _plot_errors({"GMRES residual": residuals}, "Iterative ParaDiag for ADE", output_dir / "iterative_paradiag_ade.png")
    true_residual = float(np.linalg.norm(k_matvec(solution) - b) / np.linalg.norm(b))
    return _save_metrics(output_dir, "iterative_paradiag_ade", {
        "parameters": {"Nx": nx, "Nt": nt, "nu": nu, "alpha": alpha},
        "gmres_info": int(info), "iterations": len(residuals),
        "reported_residuals": residuals, "true_relative_residual": true_residual,
    })


def run_stmg_linear(output_dir, quick=False):
    output_dir = _prepare(output_dir)
    level = 6 if quick else 8
    nx = 2**level - 1
    nt = nx
    theta, nu, t_final = 0.5, 0.001, 5.0
    dt, dx = t_final / nt, 1 / (nx + 1)
    a = linear_ade_operator(nx, dx, nu, periodic=False).tocsc()
    eye = sparse.eye(nx, format="csc")
    q = eye + dt * theta * a
    tq = eye - dt * (1 - theta) * a
    qsolve = splu(q)
    x = np.arange(1, nx + 1) * dx
    u0 = np.sin(2 * np.pi * x)
    exact = np.zeros((nx, nt + 1))
    exact[:, 0] = u0
    for j in range(nt):
        exact[:, j + 1] = qsolve.solve(tq @ exact[:, j])
    nx_c = (nx + 1) // 2 - 1
    nt_c = (nt + 1) // 2 - 1
    px = sparse.lil_matrix((nx, nx_c))
    for j in range(nx_c):
        center = 2 * j + 1
        px[center, j] = 1.0
        px[center - 1, j] = 0.5
        px[center + 1, j] = 0.5
    px = px.tocsc()
    rx = 0.5 * px.T
    pt = sparse.lil_matrix((nt, nt_c))
    for j in range(nt_c):
        center = 2 * j + 1
        pt[center, j] = 1.0
        pt[center - 1, j] = 0.5
        pt[center + 1, j] = 0.5
    pt = pt.tocsc()
    rt = 0.5 * pt.T
    ac = (rx @ a @ px).tocsc()
    eye_c = sparse.eye(nx_c, format="csc")
    qc = eye_c + 2 * dt * theta * ac
    tqc = eye_c - 2 * dt * (1 - theta) * ac
    qcsolve = splu(qc)
    etas = np.array([0.25, 0.5, 0.75, 1.0]) if quick else np.arange(0.1, 1.1001, 0.016)
    rng = np.random.default_rng(0)
    errors_at_5 = []
    errors_at_10 = []
    errors_at_15 = []
    for eta in etas:
        u = rng.random((nx, nt + 1))
        u[:, 0] = u0
        history = [float(np.max(np.abs(u - exact)))]
        for _ in range(15):
            for _ in range(3):
                old = u.copy()
                residual = tq @ old[:, :-1] - q @ old[:, 1:]
                u[:, 1:] = old[:, 1:] + eta * qsolve.solve(residual)
            residual = tq @ u[:, :-1] - q @ u[:, 1:]
            residual_c = np.asarray(rx @ residual @ rt.T)
            error_c = np.zeros((nx_c, nt_c + 1))
            for j in range(nt_c):
                error_c[:, j + 1] = qcsolve.solve(tqc @ error_c[:, j] + residual_c[:, j])
            u[:, 1:] += np.asarray(px @ error_c[:, 1:] @ pt.T)
            for _ in range(3):
                old = u.copy()
                residual = tq @ old[:, :-1] - q @ old[:, 1:]
                u[:, 1:] = old[:, 1:] + eta * qsolve.solve(residual)
            history.append(float(np.max(np.abs(u - exact))))
        errors_at_5.append(history[5])
        errors_at_10.append(history[10])
        errors_at_15.append(history[15])
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.semilogy(etas, errors_at_5, label="k=5")
    ax.semilogy(etas, errors_at_10, label="k=10")
    ax.semilogy(etas, errors_at_15, label="k=15")
    ax.axvline(0.5, color=COLORS["navy"], ls="--", lw=1.25, label=r"$\eta=1/2$")
    ax.set(
        xlabel=r"Damping parameter $\eta$",
        ylabel=r"Maximum error $\|e^k\|_\infty$",
        title="STMG block-Jacobi damping study",
    )
    style_axis(ax)
    ax.legend()
    save_figure(fig, output_dir / "stmg_heat_ade.png")
    best = int(np.argmin(errors_at_15))
    return _save_metrics(output_dir, "stmg_heat_ade", {
        "parameters": {"Nx": nx, "Nt": nt, "nu": nu},
        "etas": etas.tolist(), "error_k5": errors_at_5,
        "error_k10": errors_at_10, "error_k15": errors_at_15,
        "best_eta_k15": float(etas[best]), "best_error_k15": float(errors_at_15[best]),
        "note": "two-grid STMG with three pre- and three post-smoothing steps",
    })
