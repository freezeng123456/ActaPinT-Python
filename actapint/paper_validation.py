from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu

from .gpu import CupyBurgersFineStepper, backend_metadata, resolve_backend
from .integrators import burgers_fine_step, burgers_ie_step, linear_step_matrix, repeated_linear_step
from .operators import burgers_operators, linear_ade_operator
from .plotting import COLORS, save_figure, style_axis


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_value),
        encoding="utf-8",
    )


def _case_title(label: str) -> str:
    if label == "Heat":
        return "Heat"
    return label.replace("ADE nu=", r"ADE, $\nu=") + "$"


def _sdirk22_stability(z):
    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    return (1.0 - (1.0 - 2.0 * gamma) * z) / (1.0 + gamma * z) ** 2


def _linear_propagators(n, dx, nu, coarse_dt, substeps, periodic):
    a = linear_ade_operator(n, dx, nu, periodic=periodic)
    fine_dt = coarse_dt / substeps
    fine = repeated_linear_step(a, fine_dt, substeps, "sdirk22")
    coarse = linear_step_matrix(a, coarse_dt, "ie")
    eig = np.linalg.eigvals(coarse_dt * a.toarray())
    rf = _sdirk22_stability(eig / substeps) ** substeps
    rg = 1.0 / (1.0 + eig)
    numerator = np.abs(rg - rf)
    denominator = 1.0 - np.abs(rg)
    rho_p = np.zeros_like(numerator, dtype=float)
    active = denominator > 1e-12
    rho_p[active] = numerator[active] / denominator[active]
    # The periodic constant mode has F=G=1 and contributes zero error.
    rho_p[(~active) & (numerator > 1e-10)] = np.inf
    rho_m = np.abs(rf) * rho_p
    return (
        fine,
        coarse,
        float(np.max(rho_p)),
        float(np.max(rho_m)),
        eig / coarse_dt,
        rho_p,
        rho_m,
    )


def _reference(fine, u0, nt):
    result = np.zeros((u0.size, nt + 1))
    result[:, 0] = u0
    for j in range(nt):
        result[:, j + 1] = fine @ result[:, j]
    return result


def _parareal_linear(fine, coarse, reference, initial, iterations):
    current = initial.copy()
    coarse_old = current.copy()
    errors = [float(np.max(np.abs(current - reference)))]
    for _ in range(iterations):
        fine_values = np.column_stack([fine @ current[:, j] for j in range(reference.shape[1] - 1)])
        updated = current.copy()
        for j in range(reference.shape[1] - 1):
            new_coarse = coarse @ updated[:, j]
            updated[:, j + 1] = new_coarse + fine_values[:, j] - coarse_old[:, j + 1]
            coarse_old[:, j + 1] = new_coarse
        current = updated
        errors.append(float(np.max(np.abs(current - reference))))
    return errors


def _mgrit_fcf(fine, coarse, reference, initial, iterations):
    nt = reference.shape[1] - 1
    current = initial.copy()
    current[:, 1] = fine @ reference[:, 0]
    errors = [float(np.max(np.abs(current - reference)))]
    for _ in range(iterations):
        updated = current.copy()
        f2 = np.zeros_like(current)
        g = np.zeros_like(current)
        for j in range(1, nt - 1):
            mid = fine @ current[:, j - 1]
            f2[:, j + 1] = fine @ mid
            g[:, j + 1] = coarse @ mid
        for j in range(1, nt - 1):
            updated[:, j + 1] = coarse @ updated[:, j] + f2[:, j + 1] - g[:, j + 1]
        current = updated
        errors.append(float(np.max(np.abs(current[:, 2:nt] - reference[:, 2:nt]))))
    return errors


def validate_figure_45(output_dir, quick=False, backend="auto"):
    """Parareal diffusion-parameter sweep for ADE and Burgers (paper Figure 4.5)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_backend = resolve_backend(backend)
    n = 64 if quick else 128
    nt = 20 if quick else 40
    substeps = 8 if quick else 32
    t_final = 2.0 if quick else 4.0
    coarse_dt = t_final / nt
    dx = 1.0 / n
    nus = [1.0, 0.1, 0.02]
    linear_results = {}
    burgers_results = {}

    for nu in nus:
        fine, coarse, rho_p, _, _, _, _ = _linear_propagators(
            n, dx, nu, coarse_dt, substeps, True
        )
        x = np.linspace(0, 1, n)
        u0 = np.sin(2 * np.pi * x) ** 2
        ref = _reference(fine, u0, nt)
        initial = np.tile(u0[:, None], (1, nt + 1))
        errors = _parareal_linear(fine, coarse, ref, initial, nt + 1)
        linear_results[str(nu)] = {
            "rho_l_max": rho_p,
            "errors": errors,
            "iteration_to_1e-10": next((i for i, e in enumerate(errors) if e <= 1e-10), None),
        }

        a1, a2 = burgers_operators(n, dx, nu, periodic=True)
        fine_b = lambda y: burgers_fine_step(
            a1, a2, y, coarse_dt / substeps, substeps, "sdirk22", 1e-12
        )
        fine_b_batch = (
            CupyBurgersFineStepper(
                a1,
                a2,
                coarse_dt / substeps,
                substeps,
                tol=1e-14,
            )
            if selected_backend == "gpu"
            else None
        )
        coarse_b = lambda y: burgers_ie_step(a1, a2, coarse_dt, y, 1e-12)
        ref_b = np.zeros((n, nt + 1))
        ref_b[:, 0] = u0
        for j in range(nt):
            ref_b[:, j + 1] = fine_b(ref_b[:, j])
        current = np.tile(u0[:, None], (1, nt + 1))
        coarse_old = current.copy()
        errors_b = [float(np.max(np.abs(current - ref_b)))]
        for _ in range(nt + 1):
            fvals = (
                fine_b_batch(current[:, :nt])
                if fine_b_batch is not None
                else np.column_stack([fine_b(current[:, j]) for j in range(nt)])
            )
            updated = current.copy()
            for j in range(nt):
                cg = coarse_b(updated[:, j])
                updated[:, j + 1] = cg + fvals[:, j] - coarse_old[:, j + 1]
                coarse_old[:, j + 1] = cg
            current = updated
            errors_b.append(float(np.max(np.abs(current - ref_b))))
        burgers_results[str(nu)] = {
            "errors": errors_b,
            "iteration_to_1e-10": next((i for i, e in enumerate(errors_b) if e <= 1e-10), None),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    markers = ("o", "s", "^")
    for marker, nu in zip(markers, nus):
        axes[0].semilogy(
            linear_results[str(nu)]["errors"],
            marker=marker,
            markevery=3,
            label=fr"$\nu={nu:g}$",
        )
        axes[1].semilogy(
            burgers_results[str(nu)]["errors"],
            marker=marker,
            markevery=3,
            label=fr"$\nu={nu:g}$",
        )
    axes[0].set_title("ADE: Parareal")
    axes[1].set_title("Burgers: Parareal")
    for ax in axes:
        ax.set(xlabel="Iteration", ylabel=r"Maximum error $\|e^k\|_\infty$")
        style_axis(ax)
        ax.legend()
        ax.set_ylim(1e-14, 2)
    fig.suptitle(
        "Diffusion controls Parareal convergence",
        color=COLORS["navy"],
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure_4_5_validation.png")
    data = {
        "paper_figure": "4.5",
        "parameters": {"Nx": n, "Nt": nt, "J": substeps, "T": t_final, "dx": dx},
        "compute": backend_metadata(selected_backend),
        "ade": linear_results,
        "burgers": burgers_results,
        "expected": "Convergence deteriorates as nu decreases.",
    }
    _write_json(output_dir / "figure_4_5_validation.json", data)
    return data


def validate_figure_410(output_dir, quick=False):
    """Fair-cost Parareal/MGRiT comparison (paper Figures 4.9-4.10)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 80 if quick else 160
    nt = 20 if quick else 40
    substeps = 10 if quick else 20
    t_final = 2.5 if quick else 5.0
    coarse_dt = t_final / nt
    dx = 1.0 / n
    cases = [("Heat", -1.0, False), ("ADE nu=0.1", 0.1, True),
             ("ADE nu=0.01", 0.01, True), ("ADE nu=0.002", 0.002, True)]
    rng = np.random.default_rng(0)
    results = {}

    for label, nu, periodic in cases:
        fine, coarse, rho_p, rho_m, eig, rho_p_modes, rho_m_modes = _linear_propagators(
            n, dx, nu, coarse_dt, substeps, periodic
        )
        x = np.linspace(0, 1, n)
        u0 = np.sin(8 * np.pi * (1 - x) ** 2) ** 2
        ref = _reference(fine, u0, nt)
        initial = rng.random((n, nt + 1))
        initial[:, 0] = u0
        parareal = _parareal_linear(fine, coarse, ref, initial, min(2 * 25, nt + 1))
        mgrit = _mgrit_fcf(fine, coarse, ref, initial, 25)
        results[label] = {
            "nu": nu,
            "rho_l_parareal": rho_p,
            "rho_l_mgrit": rho_m,
            "parareal_errors": parareal,
            "parareal_equal_cost_errors": parareal[::2],
            "mgrit_errors": mgrit,
            "spectrum_real": eig.real,
            "spectrum_imag": eig.imag,
            "rho_l_parareal_modes": rho_p_modes,
            "rho_l_mgrit_modes": rho_m_modes,
            "paper_prediction": "divergent" if rho_p > 1 or rho_m > 1 else "convergent",
        }

    fig, axes = plt.subplots(
        2, 2, figsize=(11.8, 8.8), constrained_layout=True, sharex=True
    )
    for ax, (label, _, _) in zip(axes.ravel(), cases):
        result = results[label]
        ax.semilogy(
            result["parareal_equal_cost_errors"],
            "o--",
            markevery=2,
            label="Two Parareal iterations",
        )
        ax.semilogy(
            result["mgrit_errors"],
            "s-",
            markevery=2,
            label="One FCF-MGRiT iteration",
        )
        ax.axhline(
            max((coarse_dt / substeps) ** 2, dx**2),
            color=COLORS["navy"],
            ls=":",
            lw=1.4,
            label="Discretization scale",
        )
        ax.set_title(
            f"{_case_title(label)}\n"
            fr"$\rho_P={result['rho_l_parareal']:.3f},\ "
            fr"\rho_M={result['rho_l_mgrit']:.3f}$"
        )
        ax.set(xlabel="Equal-cost iteration", ylabel=r"Maximum error $\|e^k\|_\infty$")
        style_axis(ax)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Measured Parareal and MGRiT convergence at equal fine-solve cost",
        color=COLORS["navy"],
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure_4_10_validation.png")

    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.8), constrained_layout=True)
    normalization = Normalize(vmin=0.0, vmax=1.5)
    scatter = None
    for column, (label, _, _) in enumerate(cases):
        result = results[label]
        for row, (method, values, vmax) in enumerate(
            (
                ("Parareal", result["rho_l_parareal_modes"], result["rho_l_parareal"]),
                ("FCF-MGRiT", result["rho_l_mgrit_modes"], result["rho_l_mgrit"]),
            )
        ):
            ax = axes[row, column]
            values = np.asarray(values)
            scatter = ax.scatter(
                result["spectrum_real"],
                result["spectrum_imag"],
                c=values,
                cmap="viridis",
                norm=normalization,
                s=16 + 54 * np.minimum(values / 1.5, 1.0),
                alpha=0.9,
                linewidths=0.25,
                edgecolors=COLORS["paper"],
            )
            ax.set_title(
                f"{_case_title(label)}\n"
                fr"$\rho_{{\max}}={vmax:.3f}$"
            )
            ax.set_xlabel(r"$\operatorname{Re}\lambda(A)$")
            if column == 0:
                ax.set_ylabel(f"{method}\n" + r"$\operatorname{Im}\lambda(A)$")
            else:
                ax.set_ylabel(r"$\operatorname{Im}\lambda(A)$")
            style_axis(ax)
    colorbar = fig.colorbar(
        scatter,
        ax=axes,
        shrink=0.84,
        pad=0.015,
        aspect=32,
        label=r"Long-time convergence factor $\rho_\ell$",
    )
    colorbar.outline.set_edgecolor(COLORS["grid"])
    fig.suptitle(
        "Mode-wise long-time convergence factors",
        color=COLORS["navy"],
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure_4_9_validation.png")
    data = {
        "paper_figures": ["4.9", "4.10"],
        "parameters": {"Nx": n, "Nt": nt, "J": substeps, "T": t_final, "DeltaT": coarse_dt},
        "results": results,
        "expected": (
            "At equal fine-solve cost, one FCF-MGRiT iteration behaves similarly to "
            "two Parareal iterations when the coarse solver is accurate; both deteriorate "
            "and eventually diverge as nu approaches the hyperbolic regime."
        ),
    }
    _write_json(output_dir / "figure_4_10_validation.json", data)
    return data


def _paradiag_special_eigenvalues(nu, nx, nt, t_final, alpha=1.0):
    """Non-unit eigenvalues of P^{-1}K for backward-Euler ParaDiag-II.

    For each spatial eigenmode, the temporal preconditioner differs from the
    all-at-once matrix by a rank-one wrap-around term. Hence ``nt - 1``
    eigenvalues are exactly one, and the remaining eigenvalue has the closed
    form below. This avoids building an ``(nx*nt)``-square dense matrix.
    """
    dt = t_final / nt
    operator = linear_ade_operator(nx, 1 / nx, nu, periodic=False).toarray()
    spatial_eigenvalues = np.linalg.eigvals(operator)
    r1 = 1.0 + dt * spatial_eigenvalues
    r2 = -1.0
    q = alpha * (-r2 / r1) ** nt
    return 1.0 / (1.0 - q)


def _paradiag_ade_case(nu, nx, nt, t_final):
    dt = t_final / nt
    a = linear_ade_operator(nx, 1 / nx, nu, periodic=False).tocsc()
    eye = sparse.eye(nx, format="csc")
    r1, r2 = eye + dt * a, -eye
    x = np.linspace(0, 1, nx)
    u0 = np.sin(2 * np.pi * x)
    b = np.zeros(nx * nt)
    b[:nx] = -r2 @ u0

    def kmat(vector):
        u = vector.reshape((nx, nt), order="F")
        z = np.empty_like(u)
        z[:, 0] = r1 @ u[:, 0]
        for j in range(1, nt):
            z[:, j] = r1 @ u[:, j] + r2 @ u[:, j - 1]
        return z.reshape(-1, order="F")

    c = np.zeros(nt)
    c[1] = 1.0
    eig_c = np.fft.fft(c)
    solvers = [splu((r1 + eig_c[j] * r2).tocsc()) for j in range(nt)]

    def psolve(vector):
        u = vector.reshape((nx, nt), order="F")
        transformed = np.fft.fft(u.T, axis=0).T
        solved = np.column_stack([solvers[j].solve(transformed[:, j]) for j in range(nt)])
        return np.real_if_close(np.fft.ifft(solved.T, axis=0).T).reshape(-1, order="F")

    residuals = []
    operator = LinearOperator((nx * nt, nx * nt), matvec=kmat, dtype=float)
    preconditioner = LinearOperator((nx * nt, nx * nt), matvec=psolve, dtype=float)
    solution, info = gmres(
        operator, b, x0=np.tile(0.125 * u0, nt), M=preconditioner,
        rtol=1e-12, atol=0.0, restart=min(100, nt), maxiter=1000,
        callback=lambda value: residuals.append(float(value)), callback_type="pr_norm",
    )
    true_residual = float(np.linalg.norm(kmat(solution) - b) / np.linalg.norm(b))
    spectrum = _paradiag_special_eigenvalues(nu, nx, nt, t_final)
    return {
        "nu": nu,
        "iterations": len(residuals),
        "residuals": residuals,
        "true_relative_residual": true_residual,
        "gmres_info": int(info),
        "spectrum_nonunit_real": spectrum.real,
        "spectrum_nonunit_imag": spectrum.imag,
        "unit_eigenvalue_multiplicity": nx * (nt - 1),
    }


def _paradiag_wave_spectrum(nx, nt, dt, alpha, gamma):
    """Return the two non-unit eigenvalues per spatial wave mode."""
    spatial = linear_ade_operator(nx, 1 / nx, -1.0, periodic=False).toarray()
    spatial_eigenvalues = np.linalg.eigvalsh(spatial)
    shift = sparse.diags(np.ones(nt - 1), -1, shape=(nt, nt), format="lil")
    shift[0, -1] = alpha
    shift = shift.tocsc()
    identity_t = sparse.eye(nt, format="csc")
    shift_square = shift @ shift
    basis = np.zeros((nt, 2))
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    special = []
    for eigenvalue in spatial_eigenvalues:
        z = dt**2 * eigenvalue
        r1 = 1.0 + z / 12.0 + 10.0 * gamma * z**2 / 12.0
        r2 = -2.0 + 10.0 * z / 12.0 - 20.0 * gamma * z**2 / 12.0
        preconditioner = (
            r1 * (identity_t + shift_square) + r2 * shift
        ).tocsc()
        pinv_e = splu(preconditioner).solve(basis)
        vt = np.zeros((2, nt))
        vt[0, -2] = alpha * r1
        vt[0, -1] = alpha * r2
        vt[1, -1] = alpha * r1
        special.extend(np.linalg.eigvals(np.eye(2) - vt @ pinv_e))
    return np.asarray(special)


def _paradiag_wave_case(quick=False):
    """Wave-equation part of paper Figure 3.15(e,f)."""
    nx = 50 if quick else 100
    dt = 1 / 25 if quick else 1 / 50
    t_final = 2.0
    nt = int(round(t_final / dt)) - 1
    alpha = 1.0
    gamma = 1 / 100
    dx = 1 / nx

    laplacian = linear_ade_operator(nx, dx, -1.0, periodic=False).tocsc()
    identity_x = sparse.eye(nx, format="csc")
    scaled = dt**2 * laplacian
    r1 = identity_x + scaled / 12.0 + 10.0 * gamma * (scaled @ scaled) / 12.0
    r2 = -2.0 * identity_x + 10.0 * scaled / 12.0 - 20.0 * gamma * (scaled @ scaled) / 12.0

    x = np.linspace(0, 1, nx)
    u0 = np.sin(2 * np.pi * x)
    u1 = splu((identity_x + 0.5 * dt**2 * laplacian).tocsc()).solve(u0)
    b = np.zeros(nx * nt)
    b[:nx] = -r2 @ u1 - r1 @ u0
    b[nx:2 * nx] = -r1 @ u1

    def kmat(vector):
        u = vector.reshape((nx, nt), order="F")
        z = np.empty_like(u)
        z[:, 0] = r1 @ u[:, 0]
        z[:, 1] = r1 @ u[:, 1] + r2 @ u[:, 0]
        for j in range(2, nt):
            z[:, j] = r1 @ u[:, j] + r2 @ u[:, j - 1] + r1 @ u[:, j - 2]
        return z.reshape(-1, order="F")

    da = alpha ** (np.arange(nt) / nt)
    inv_da = 1.0 / da
    c = np.zeros(nt)
    c[1] = 1.0
    tc = np.zeros(nt)
    tc[0] = 1.0
    tc[2] = 1.0
    eig_c = np.fft.fft(da * c)
    eig_tc = np.fft.fft(da * tc)
    solvers = [
        splu((eig_tc[j] * r1 + eig_c[j] * r2).tocsc())
        for j in range(nt)
    ]

    def psolve(vector):
        u = vector.reshape((nx, nt), order="F")
        transformed = np.fft.fft(da[:, None] * u.T, axis=0).T
        solved = np.column_stack(
            [solvers[j].solve(transformed[:, j]) for j in range(nt)]
        )
        result = (inv_da[:, None] * np.fft.ifft(solved.T, axis=0)).T
        return result.real.reshape(-1, order="F")

    residuals = []
    operator = LinearOperator((nx * nt, nx * nt), matvec=kmat, dtype=float)
    preconditioner = LinearOperator((nx * nt, nx * nt), matvec=psolve, dtype=float)
    solution, info = gmres(
        operator,
        b,
        x0=np.tile(0.125 * u0, nt),
        M=preconditioner,
        rtol=1e-12,
        atol=0.0,
        restart=min(150, nt),
        maxiter=1000,
        callback=lambda value: residuals.append(float(value)),
        callback_type="pr_norm",
    )
    true_residual = float(np.linalg.norm(kmat(solution) - b) / np.linalg.norm(b))
    spectrum = _paradiag_wave_spectrum(nx, nt, dt, alpha, gamma)
    iteration_to_preconditioned_1e_11 = next(
        (index for index, value in enumerate(residuals, start=1) if value <= 1e-11),
        None,
    )
    return {
        "parameters": {
            "Nx": nx,
            "Nt": nt,
            "dt": dt,
            "T": t_final,
            "alpha": alpha,
            "gamma": gamma,
        },
        "iterations": len(residuals),
        "iteration_to_preconditioned_residual_1e-11": iteration_to_preconditioned_1e_11,
        "residuals": residuals,
        "true_relative_residual": true_residual,
        "gmres_info": int(info),
        "spectrum_nonunit_real": spectrum.real,
        "spectrum_nonunit_imag": spectrum.imag,
        "unit_eigenvalue_multiplicity": nx * (nt - 2),
    }


def validate_figure_315(output_dir, quick=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nx = 50 if quick else 100
    nt = 50 if quick else 100
    heat = _paradiag_ade_case(-1.0, nx, nt, 2.0)
    results = {str(nu): _paradiag_ade_case(nu, nx, nt, 2.0) for nu in (1e-3, 1e-6)}
    wave = _paradiag_wave_case(quick)

    fig, axes = plt.subplots(3, 2, figsize=(11.8, 11.2), constrained_layout=True)
    heat_spectrum = (
        np.asarray(heat["spectrum_nonunit_real"])
        + 1j * np.asarray(heat["spectrum_nonunit_imag"])
    )
    axes[0, 0].scatter(
        heat_spectrum.real - 1.0,
        heat_spectrum.imag,
        s=25,
        marker="+",
        color=COLORS["blue"],
    )
    axes[0, 0].scatter([0.0], [0.0], s=40, marker="x", color=COLORS["navy"])
    axes[0, 0].set(
        xlabel=r"$\operatorname{Re}\{\lambda(P^{-1}K)-1\}$",
        ylabel=r"$\operatorname{Im}\lambda(P^{-1}K)$",
        title="Heat equation · preconditioned spectrum",
    )
    axes[0, 1].semilogy(
        np.arange(1, len(heat["residuals"]) + 1),
        heat["residuals"],
        "o-",
        color=COLORS["blue"],
        label="Heat",
    )
    axes[0, 1].set(
        xlabel="GMRES iteration",
        ylabel="Preconditioned residual",
        title="Heat equation · measured convergence",
    )

    markers = ("s", "o")
    for color, marker, nu in zip((COLORS["teal"], COLORS["rose"]), markers, (1e-3, 1e-6)):
        case = results[str(nu)]
        spectrum = (
            np.asarray(case["spectrum_nonunit_real"])
            + 1j * np.asarray(case["spectrum_nonunit_imag"])
        )
        axes[1, 0].scatter(
            spectrum.real,
            spectrum.imag,
            s=28,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            label=fr"$\nu={nu:g}$",
        )
        axes[1, 1].semilogy(
            np.arange(1, len(case["residuals"]) + 1),
            case["residuals"],
            marker=marker,
            markevery=1,
            color=color,
            label=fr"$\nu={nu:g}$",
        )
    axes[1, 0].scatter([1.0], [0.0], s=40, marker="x", color=COLORS["navy"])
    axes[1, 0].set(
        xlabel=r"$\operatorname{Re}\lambda(P^{-1}K)$",
        ylabel=r"$\operatorname{Im}\lambda(P^{-1}K)$",
        title="Advection–diffusion · preconditioned spectrum",
    )
    axes[1, 1].set(
        xlabel="GMRES iteration",
        ylabel="Preconditioned residual",
        title="Advection–diffusion · measured convergence",
    )
    wave_spectrum = (
        np.asarray(wave["spectrum_nonunit_real"])
        + 1j * np.asarray(wave["spectrum_nonunit_imag"])
    )
    axes[2, 0].scatter(
        wave_spectrum.real,
        wave_spectrum.imag,
        s=22,
        marker="+",
        color=COLORS["indigo"],
        label="Non-unit eigenvalues",
    )
    axes[2, 0].scatter(
        [1.0],
        [0.0],
        s=40,
        marker="x",
        color=COLORS["navy"],
        label="Unit cluster",
    )
    axes[2, 0].set(
        xlabel=r"$\operatorname{Re}\lambda(P^{-1}K)$",
        ylabel=r"$\operatorname{Im}\lambda(P^{-1}K)$",
        title="Wave equation · preconditioned spectrum",
    )
    axes[2, 1].semilogy(
        np.arange(1, len(wave["residuals"]) + 1),
        wave["residuals"],
        color=COLORS["indigo"],
        marker="o",
        markevery=max(1, len(wave["residuals"]) // 12),
        label="Wave",
    )
    axes[2, 1].set(
        xlabel="GMRES iteration",
        ylabel="Preconditioned residual",
        title="Wave equation · measured convergence",
    )
    for ax in axes.ravel():
        style_axis(ax)
        if ax in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]):
            ax.legend(fontsize=8)
    fig.suptitle(
        "ParaDiag-II: spectral clustering predicts Krylov convergence",
        color=COLORS["navy"],
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure_3_15_validation.png")
    data = {
        "paper_figure": "3.15(a-f)",
        "parameters": {"Nx": nx, "Nt": nt, "T": 2.0},
        "heat": heat,
        "results": results,
        "wave": wave,
        "expected": (
            "The heat spectrum clusters tightly at one. ADE requires around "
            "3 iterations for nu=1e-3 and 13 iterations for nu=1e-6. The wave "
            "problem is substantially harder and has an order-100 Krylov phase; "
            "the exact endpoint depends on the GMRES residual and stopping convention."
        ),
    }
    _write_json(output_dir / "figure_3_15_validation.json", data)
    return data


def _stmg_components(level, nu, t_final=5.0, theta=1.0):
    nx = 2**level - 1
    nt = nx
    dx = 1.0 / (nx + 1)
    dt = t_final / nt
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
    pt = sparse.lil_matrix((nt, nt_c))
    for j in range(nx_c):
        center = 2 * j + 1
        px[center, j] = 1.0
        px[center - 1, j] = px[center + 1, j] = 0.5
    for j in range(nt_c):
        center = 2 * j + 1
        pt[center, j] = 1.0
        pt[center - 1, j] = pt[center + 1, j] = 0.5
    px, pt = px.tocsc(), pt.tocsc()
    rx, rt = 0.5 * px.T, 0.5 * pt.T
    ac = (rx @ a @ px).tocsc()
    eye_c = sparse.eye(nx_c, format="csc")
    qc = eye_c + 2 * dt * theta * ac
    tqc = eye_c - 2 * dt * (1 - theta) * ac
    return {
        "nx": nx, "nt": nt, "u0": u0, "exact": exact, "q": q, "tq": tq,
        "qsolve": qsolve, "px": px, "pt": pt, "rx": rx, "rt": rt,
        "tqc": tqc, "qcsolve": splu(qc),
    }


def _stmg_history(parts, eta, smooth_steps, iterations, seed=0, matlab_residual=False):
    rng = np.random.default_rng(seed)
    u = rng.random((parts["nx"], parts["nt"] + 1))
    u[:, 0] = parts["u0"]
    history = [float(np.max(np.abs(u - parts["exact"])))]
    for _ in range(iterations):
        last_old = u
        for _ in range(smooth_steps):
            old = u.copy()
            last_old = old
            residual = parts["tq"] @ old[:, :-1] - parts["q"] @ old[:, 1:]
            u[:, 1:] = old[:, 1:] + eta * parts["qsolve"].solve(residual)
        residual_state = last_old if matlab_residual else u
        residual = (
            parts["tq"] @ residual_state[:, :-1]
            - parts["q"] @ residual_state[:, 1:]
        )
        residual_c = np.asarray(parts["rx"] @ residual @ parts["rt"].T)
        error_c = np.zeros((parts["rx"].shape[0], parts["rt"].shape[0] + 1))
        for j in range(parts["rt"].shape[0]):
            error_c[:, j + 1] = parts["qcsolve"].solve(
                parts["tqc"] @ error_c[:, j] + residual_c[:, j]
            )
        u[:, 1:] += np.asarray(parts["px"] @ error_c[:, 1:] @ parts["pt"].T)
        for _ in range(smooth_steps):
            old = u.copy()
            residual = parts["tq"] @ old[:, :-1] - parts["q"] @ old[:, 1:]
            u[:, 1:] = old[:, 1:] + eta * parts["qsolve"].solve(residual)
        history.append(float(np.max(np.abs(u - parts["exact"]))))
    return history


def validate_figures_419_420(output_dir, quick=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    level = 6 if quick else 8
    etas = np.linspace(0.15, 1.05, 19) if quick else np.arange(0.1, 1.1001, 0.016)
    scan = {}
    scan_consistent = {}
    for label, nu in (("Heat", -1.0), ("ADE nu=0.01", 0.01)):
        parts = _stmg_components(level, nu)
        curves = {5: [], 10: [], 15: []}
        curves_consistent = {5: [], 10: [], 15: []}
        for eta in etas:
            history = _stmg_history(
                parts, float(eta), smooth_steps=1, iterations=15, seed=0,
                matlab_residual=True,
            )
            history_consistent = _stmg_history(
                parts, float(eta), smooth_steps=1, iterations=15, seed=0,
                matlab_residual=False,
            )
            for k in curves:
                curves[k].append(history[k])
                curves_consistent[k].append(history_consistent[k])
        scan[label] = {
            "nu": nu,
            "errors": {str(k): curves[k] for k in curves},
            "best_eta_k15": float(etas[int(np.argmin(curves[15]))]),
            "residual_variant": "original MATLAB uold residual",
        }
        scan_consistent[label] = {
            "nu": nu,
            "errors": {str(k): curves_consistent[k] for k in curves_consistent},
            "best_eta_k15": float(etas[int(np.argmin(curves_consistent[15]))]),
            "residual_variant": "post-smoothing consistent residual",
        }

    convergence = {}
    for smooth_steps in (1, 3):
        for label, nu in (("Heat", -1.0), ("ADE nu=0.1", 0.1),
                          ("ADE nu=0.01", 0.01), ("ADE nu=0.001", 0.001)):
            parts = _stmg_components(level, nu)
            convergence[f"{smooth_steps} smooth: {label}"] = _stmg_history(
                parts, eta=0.5, smooth_steps=smooth_steps, iterations=30, seed=0,
                matlab_residual=True,
            )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    for ax, label in zip(axes, ("Heat", "ADE nu=0.01")):
        for marker, k in zip(("o", "s", "^"), (5, 10, 15)):
            ax.semilogy(
                etas,
                scan[label]["errors"][str(k)],
                marker=marker,
                markevery=max(1, len(etas) // 10),
                label=fr"$k={k}$",
            )
        ax.axvline(
            0.5,
            color=COLORS["navy"],
            ls=":",
            lw=1.4,
            label=r"$\eta=1/2$",
        )
        ax.set(
            title=label.replace("nu", r"$\nu$"),
            xlabel=r"Damping parameter $\eta$",
            ylabel=r"Maximum error $\|e^k\|_\infty$",
        )
        style_axis(ax)
        ax.legend()
    fig.suptitle(
        "STMG damping scan",
        color=COLORS["navy"],
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure_4_19_validation.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    for ax, smooth_steps in zip(axes, (1, 3)):
        for marker, label in zip(
            ("o", "s", "^", "D"),
            ("Heat", "ADE nu=0.1", "ADE nu=0.01", "ADE nu=0.001"),
        ):
            ax.semilogy(
                convergence[f"{smooth_steps} smooth: {label}"],
                marker=marker,
                markevery=3,
                label=label.replace("nu", r"$\nu$"),
            )
        ax.set(title=f"{smooth_steps} pre/post smoothing step(s)",
               xlabel="STMG iteration", ylabel=r"Maximum error $\|e^k\|_\infty$")
        style_axis(ax)
        ax.legend(fontsize=8)
        ax.set_ylim(1e-11, 2)
    fig.suptitle(
        r"STMG convergence with fixed damping $\eta=1/2$",
        color=COLORS["navy"],
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure_4_20_validation.png")
    data = {
        "paper_figures": ["4.19", "4.20"],
        "parameters": {
            "level": level,
            "eta_scan": etas,
            "theta": 1.0,
            "time_integrator": "backward Euler",
            "T": 5.0,
        },
        "damping_scan": scan,
        "damping_scan_consistent_residual": scan_consistent,
        "eta_half_convergence": convergence,
        "expected": (
            "eta=0.5 is near-optimal for one-step smoothing; three smoothing steps "
            "converge faster; convergence deteriorates as diffusion weakens."
        ),
    }
    _write_json(output_dir / "figures_4_19_4_20_validation.json", data)
    return data


def run_paper_validation(output_dir, quick=False, backend="auto"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "figure_3_15": validate_figure_315(output_dir, quick),
        "figure_4_5": validate_figure_45(output_dir, quick, backend),
        "figure_4_10": validate_figure_410(output_dir, quick),
        "figures_4_19_4_20": validate_figures_419_420(output_dir, quick),
    }
    summary = {
        "compute": results["figure_4_5"]["compute"],
        "figure_3_15": {
            nu: case["iterations"] for nu, case in results["figure_3_15"]["results"].items()
        },
        "figure_3_15_heat_iterations": results["figure_3_15"]["heat"]["iterations"],
        "figure_3_15_wave_iterations": results["figure_3_15"]["wave"]["iterations"],
        "figure_3_15_wave_iteration_to_preconditioned_residual_1e-11": (
            results["figure_3_15"]["wave"][
                "iteration_to_preconditioned_residual_1e-11"
            ]
        ),
        "figure_4_5_ade_iterations_to_1e-10": {
            nu: case["iteration_to_1e-10"] for nu, case in results["figure_4_5"]["ade"].items()
        },
        "figure_4_5_burgers_iterations_to_1e-10": {
            nu: case["iteration_to_1e-10"] for nu, case in results["figure_4_5"]["burgers"].items()
        },
        "figure_4_10_rho": {
            label: {
                "parareal": case["rho_l_parareal"],
                "mgrit": case["rho_l_mgrit"],
            }
            for label, case in results["figure_4_10"]["results"].items()
        },
        "figure_4_19_best_eta_k15": {
            label: case["best_eta_k15"]
            for label, case in results["figures_4_19_4_20"]["damping_scan"].items()
        },
    }
    _write_json(output_dir / "paper_validation_summary.json", summary)
    return output_dir / "paper_validation_summary.json"
