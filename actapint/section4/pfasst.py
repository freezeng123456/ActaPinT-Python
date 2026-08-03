"""PFASST as a two-level block iteration, for Figure 4.6.

The repository ships no script for this subsection, so the implementation follows
the paper directly: PFASST is written, as in Gander, Lunet, Ruprecht and Speck
(2023b), as the block iteration

    u^{k+1}_{n+1} = B_1^0 u^k_{n+1} + B_0^1 (chi u^{k+1}_n + dt b_n)
                                    + B_0^0 (chi u^k_n + dt b_n),

    B_1^0 = [I - T_cf phi_c^{-1} T_fc phi_f] (I - phi_f~^{-1} phi_f),
    B_0^1 = T_cf phi_c^{-1} T_fc,
    B_0^0 = [I - T_cf phi_c^{-1} T_fc phi_f] phi_f~^{-1},

where ``phi = I - dt Q kron A`` is the collocation operator of (4.10) on a block,
``phi_f~`` is the implicit Euler approximation (4.11) that makes one sweep cheap,
``T_cf`` and ``T_fc`` are the Lagrange transfers between the two node sets, and
``chi`` copies the last node of a block into every node of the next one.  Two
things are worth noting about this form: ``B_0^1`` is the only operator applied to
the *new* iterate, so one sweep costs one coarse solve per block plus local fine
work, and the fixed point of the iteration is exactly the sequential collocation
solution, which is what the error is measured against.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from ..operators import periodic_first_derivative, periodic_negative_laplacian
from ..plotting import save_figure, style_axis
from ..section2 import source_term
from .common import finish

# Figure 4.6: T = 3, periodic ends, u(x, 0) = 0, the source (2.4) with sigma =
# 1000, dx = 1/128 and dt = 1/64, so the block grid has 192 intervals.
DX = 1.0 / 128
BLOCK_DT = 1.0 / 64
T_FINAL = 3.0
SHARPNESS = 1000.0

# Radau IIA nodes, as printed: three fine nodes and two coarse ones, both node
# sets given on [0, 1] and both preceded by tau_0 = 0 for the sweep (4.11).
FINE_NODES = ((4.0 - np.sqrt(6.0)) / 10.0, (4.0 + np.sqrt(6.0)) / 10.0, 1.0)
COARSE_NODES = (1.0 / 3.0, 1.0)

# The four curves of Figure 4.6: the heat equation and the advection-diffusion
# equation with three diffusion parameters.
PROBLEMS = (
    ("heat equation", 1.0, 0.0),
    (r"ADE with $\nu=0.1$", 0.1, 1.0),
    (r"ADE with $\nu=0.001$", 1e-3, 1.0),
    (r"ADE with $\nu=0.0001$", 1e-4, 1.0),
)
ITERATIONS = 300
TOLERANCE = 1e-12
TRANSIENT = 10

# The figure prints no numbers, so the only targets are the published curves
# themselves: these are their asymptotic factor per iteration and the iteration
# at which they cross ``TOLERANCE``, digitized from the raster of the article
# page at 600 dpi, once from the marker centres and once from the drawn line.
PUBLISHED_FACTOR = {
    "heat equation": 0.7355,
    r"ADE with $\nu=0.1$": 0.7381,
    r"ADE with $\nu=0.001$": 0.7894,
    r"ADE with $\nu=0.0001$": 0.9114,
}
PUBLISHED_ITERATIONS = {
    "heat equation": 88,
    r"ADE with $\nu=0.1$": 90,
    r"ADE with $\nu=0.001$": 115,
    r"ADE with $\nu=0.0001$": 288,
}

CURVE_STYLES = (
    ("#D34FD3", "s"),
    ("#E8384F", "o"),
    ("#2F6BFF", "+"),
    ("#17324D", "*"),
)


def quadrature_matrix(nodes):
    """``Q`` of (4.10): ``Q[m, j] = int_0^{tau_m} L_j``, the collocation weights.

    Built from the nodes themselves rather than tabulated, which reproduces the
    two matrices the paper prints for Radau IIA with three and two nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    count = nodes.size
    matrix = np.empty((count, count))
    for j in range(count):
        others = np.delete(nodes, j)
        # The Lagrange basis polynomial, integrated exactly through its
        # coefficients: numpy stores them in descending powers.
        basis = np.poly(others) / np.prod(nodes[j] - others)
        antiderivative = np.polyint(basis)
        matrix[:, j] = np.polyval(antiderivative, nodes)
    return matrix


def transfer_matrix(from_nodes, to_nodes):
    """Lagrange interpolation from one node set to the other."""
    from_nodes = np.asarray(from_nodes, dtype=float)
    to_nodes = np.asarray(to_nodes, dtype=float)
    matrix = np.empty((to_nodes.size, from_nodes.size))
    for j in range(from_nodes.size):
        others = np.delete(from_nodes, j)
        basis = np.poly(others) / np.prod(from_nodes[j] - others)
        matrix[:, j] = np.polyval(basis, to_nodes)
    return matrix


def spatial_operator(dx, nu, advection):
    """``A`` of ``u_t = A u + g`` with periodic ends, the sign the paper uses."""
    n = int(round(1.0 / dx))
    operator = -nu * periodic_negative_laplacian(n, dx, 1.0)
    if advection:
        operator = operator - periodic_first_derivative(n, dx, advection)
    return operator.tocsr(), np.arange(n) * dx


def _collocation_operator(nodes, block_dt, operator):
    """``phi = I - dt Q kron A``, dense because every block reuses its inverse."""
    size = operator.shape[0]
    quadrature = quadrature_matrix(nodes)
    return np.eye(len(nodes) * size) - block_dt * np.kron(
        quadrature, operator.toarray()
    )


def _sweep_operator(nodes, block_dt, operator):
    """``phi_f~`` of (4.11): implicit Euler over the fine nodes."""
    size = operator.shape[0]
    count = len(nodes)
    difference = np.eye(count) - np.eye(count, k=-1)
    gaps = np.diff(np.concatenate([[0.0], nodes]))
    return np.kron(difference, np.eye(size)) - block_dt * np.kron(
        np.diag(gaps), operator.toarray()
    )


def block_operators(nu, advection, dx=DX, block_dt=BLOCK_DT):
    """The three block matrices of PFASST, plus the pieces the sweep needs."""
    operator, x = spatial_operator(dx, nu, advection)
    size = operator.shape[0]
    fine = _collocation_operator(FINE_NODES, block_dt, operator)
    coarse = _collocation_operator(COARSE_NODES, block_dt, operator)
    approximate = _sweep_operator(FINE_NODES, block_dt, operator)

    identity = np.eye(len(FINE_NODES) * size)
    eye = np.eye(size)
    to_fine = np.kron(transfer_matrix(COARSE_NODES, FINE_NODES), eye)
    to_coarse = np.kron(transfer_matrix(FINE_NODES, COARSE_NODES), eye)

    correction = to_fine @ np.linalg.solve(coarse, to_coarse)
    residual = identity - correction @ fine
    sweep = np.linalg.solve(approximate, identity)
    b_1_0 = residual @ (identity - sweep @ fine)
    b_0_1 = correction
    b_0_0 = residual @ sweep
    return {
        "b_1_0": b_1_0,
        "b_0_1": b_0_1,
        "b_0_0": b_0_0,
        "fine": fine,
        "size": size,
        "x": x,
    }


def _sources(x, block_dt, steps, sharpness=SHARPNESS):
    """``dt b_n`` of the iteration: the quadrature of the source over a block."""
    quadrature = quadrature_matrix(FINE_NODES)
    times = (np.arange(steps)[:, None] + np.asarray(FINE_NODES)[None, :]) * block_dt
    values = np.array(
        [
            [source_term(x, time, sharpness) for time in row]
            for row in times
        ]
    )
    return block_dt * np.einsum("mj,njx->nmx", quadrature, values).reshape(
        steps, -1
    )


def _starting_iterate(steps, width):
    """``random('unif', 0, 1, M_f N_x, N_t)`` after ``rng('default')``.

    The scripts of this chapter all start their iterations from a uniform random
    field, and Figure 4.6 needs one too: with a zero starting iterate every curve
    would begin at the size of the solution itself, about ``0.15`` for the heat
    equation, and the published curves all begin above one instead.  The draw is
    MATLAB's so that the whole reproduction uses a single convention.
    """
    stream = np.random.RandomState(5489)
    draw = stream.random_sample(width * steps)
    return draw.reshape((width, steps), order="F").T.copy()


def _collapse(matrix, size):
    """``matrix @ chi`` reduced to the block that multiplies the copied node."""
    blocks = matrix.shape[1] // size
    return sum(matrix[:, m * size : (m + 1) * size] for m in range(blocks))


def _sequential(fine, sources, size):
    """The collocation solution the iteration converges to, block by block."""
    steps = sources.shape[0]
    solution = np.zeros((steps, sources.shape[1]))
    previous = np.zeros(size)
    for n in range(steps):
        right = np.tile(previous, len(FINE_NODES)) + sources[n]
        solution[n] = np.linalg.solve(fine, right)
        previous = solution[n, -size:]
    return solution


def pfasst(operators, sources, *, iterations=ITERATIONS, tolerance=TOLERANCE):
    """Sweep the block iteration and return the error against its fixed point.

    The ``B_0^1`` term is the only one that sees the new iterate, and it only
    sees it through the last node of the previous block, so it is applied as a
    single ``M_f N_x`` by ``N_x`` matrix; everything else is assembled for all
    blocks at once, which is what makes 300 sweeps over 192 blocks cheap.
    """
    size = operators["size"]
    steps = sources.shape[0]
    reference = _sequential(operators["fine"], sources, size)

    # ``chi`` copies the last node of a block into all ``M_f`` nodes of the next,
    # so ``B chi`` acts on that one value through the sum of the column blocks of
    # ``B``.  Collapsing it here is what keeps the sequential part of the sweep to
    # one small matrix-vector product per block.
    coupling = _collapse(operators["b_0_1"], size)
    old_coupling = _collapse(operators["b_0_0"], size)
    constant = sources @ (operators["b_0_1"] + operators["b_0_0"]).T

    current = _starting_iterate(steps, sources.shape[1])
    errors = [float(np.abs(current - reference).max())]
    for _ in range(iterations):
        previous_tail = np.zeros((steps, size))
        previous_tail[1:] = current[:-1, -size:]
        fixed = current @ operators["b_1_0"].T + previous_tail @ old_coupling.T
        fixed += constant
        updated = np.empty_like(current)
        tail = np.zeros(size)
        for n in range(steps):
            updated[n] = fixed[n] + coupling @ tail
            tail = updated[n, -size:]
        current = updated
        errors.append(float(np.abs(current - reference).max()))
        if errors[-1] <= tolerance:
            break
    return errors


def run_figure_4_6(output_dir, quick=False):
    """Figure 4.6: PFASST loses its rate exactly where Parareal and MGRiT do.

    Every curve is the error of the block iteration against the collocation
    solution it converges to.  The heat equation and the diffusion-dominated ADE
    need under a hundred sweeps to reach 1e-12, while at ``nu = 1e-4`` the same
    accuracy takes about three hundred: the deterioration with weak diffusion
    that the whole section is about, now for a method whose coarse level differs
    from the fine one only in the number of collocation nodes.
    """
    problems = PROBLEMS[:2] if quick else PROBLEMS
    iterations = 60 if quick else ITERATIONS

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    curves = {}
    for (label, nu, advection), (colour, marker) in zip(problems, CURVE_STYLES):
        operators = block_operators(nu, advection)
        steps = int(round(T_FINAL / BLOCK_DT))
        sources = _sources(operators["x"], BLOCK_DT, steps)
        errors = pfasst(operators, sources, iterations=iterations)
        ax.semilogy(
            np.arange(len(errors)), errors, "-.", color=colour, marker=marker,
            markersize=6, markerfacecolor="none", markevery=10, linewidth=0.9,
            label=label,
        )
        curves[label] = {
            "nu": nu,
            "advection": advection,
            "errors": errors,
            "iterations_to_tolerance": len(errors) - 1,
            "published_iterations_to_tolerance": PUBLISHED_ITERATIONS[label],
            "measured_factor": _measured_factor(errors),
            "published_factor": PUBLISHED_FACTOR[label],
        }
    ax.set_xlabel("Iteration Index: $k$")
    ax.set_ylabel("Error")
    ax.set_xlim(0, iterations)
    ax.set_ylim(TOLERANCE, 3.0)
    style_axis(ax)
    ax.legend(fontsize=9)
    artifacts = save_figure(fig, f"{output_dir}/figure_4_6_pfasst")

    return finish(
        output_dir,
        "figure_4_6_pfasst",
        "Measured error of PFASST for the heat equation and the "
        "advection-diffusion equation with three diffusion parameters",
        {
            "parameters": {
                "dx": DX,
                "block_dt": BLOCK_DT,
                "t_final": T_FINAL,
                "blocks": int(round(T_FINAL / BLOCK_DT)),
                "boundary_condition": "periodic",
                "initial_condition": "u(x, 0) = 0",
                "source": "equation (2.4) with sigma = 1000",
                "fine_nodes": list(FINE_NODES),
                "coarse_nodes": list(COARSE_NODES),
                "quadrature": "Radau IIA on both levels",
                "starting_iterate": "random('unif', 0, 1) after rng('default')",
                "tolerance": TOLERANCE,
            },
            "curves": curves,
        },
        artifacts,
    )


def _measured_factor(errors, skip=TRANSIENT):
    """Asymptotic factor per iteration: the slope of the decaying tail.

    The first sweeps are a transient whose size depends on how the starting
    iterate is spread over the spatial modes, so a mean taken from the first
    iteration reports the transient as much as the rate.  The published curves
    are straight from about ``skip`` onwards, which is where the fit starts.
    """
    usable = np.array([value for value in errors if value > TOLERANCE])
    if usable.size < skip + 4:
        return None
    slope = np.polyfit(
        np.arange(skip, usable.size), np.log10(usable[skip:]), 1
    )[0]
    return float(10.0**slope)
