import numpy as np
import pytest
from scipy import sparse
from actapint import nka_audit as audit
from actapint.section3 import paradiag_bvm as p


@pytest.mark.parametrize("variant", ["average", "nka"])
def test_quasi_newton_update_matches_global_sparse_system(variant):
    a, d, x = p.burgers_operators(0.02, 0.1)
    u0 = p.initial_condition(x)
    steps = 8
    dt = 0.04
    u = np.random.default_rng(p.RANDOM_SEED).uniform(-1, 1, (steps, len(x)))
    b = np.zeros_like(u)
    b[0] = u0 / (2 * dt)
    time = p.bvm_matrix(steps, dt)
    weights = np.ones(steps) if variant == "average" else p.nka_weights(0.02, dt, steps)
    global_update = audit.correction(
        b, u, time, sparse.csc_matrix(a), sparse.csc_matrix(d), weights, method="global"
    )
    diagonal_update = audit.correction(
        b, u, time, sparse.csc_matrix(a), sparse.csc_matrix(d), weights
    )
    assert diagonal_update == pytest.approx(global_update, rel=1e-10, abs=1e-11)
    legacy, _, _ = p.quasi_newton(
        a,
        d,
        u0,
        steps,
        dt,
        variant=variant,
        scaling=weights,
        iterations=1,
        start="symmetric",
    )
    assert legacy == pytest.approx(u + global_update, rel=1e-10, abs=1e-11)


def test_frobenius_weights_satisfy_the_normal_equations():
    nu, dt, steps = 0.02, 0.01, 20
    a, d, x = p.burgers_operators(nu, p.NKA_COARSE_DX)
    trajectory, _ = p.trapezoidal_steps(a, d, p.initial_condition(x), steps, dt)
    _, jac = p.residual_operator(a, d)
    blocks = np.array([jac(row) for row in trajectory[1:]])
    mean = blocks.mean(axis=0)
    weights = p.nka_weights(nu, dt, steps)
    for weight, block in zip(weights, blocks):
        # Orthogonality is the optimality condition for (3.47), independently
        # of any figure markers or the particular quotient implementation.
        assert np.sum((block - weight * mean) * mean) == pytest.approx(0.0, abs=1e-10)
        optimal = np.linalg.norm(block - weight * mean)
        assert optimal < np.linalg.norm(block - (weight + 0.01) * mean)
        assert optimal < np.linalg.norm(block - (weight - 0.01) * mean)


def test_newton_reports_the_returned_iterate_residual_at_iteration_limit(monkeypatch):
    monkeypatch.setattr(p, "MAX_ITERATIONS", 1)
    a, d, x = p.burgers_operators(0.02, 0.1)
    steps, dt = 8, 0.04
    solution, reported = p.all_at_once_solution(a, d, p.initial_condition(x), steps, dt)
    rhs = np.zeros_like(solution)
    rhs[0] = p.initial_condition(x) / (2 * dt)
    residual = (
        rhs
        - p.bvm_matrix(steps, dt) @ solution
        - (a @ solution.T**2 + d @ solution.T).T
    )
    assert reported == pytest.approx(np.abs(residual).max(), abs=1e-13)
