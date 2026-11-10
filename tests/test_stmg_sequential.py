import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from actapint.stmg_fem import space_hierarchy
from actapint.stmg_sequential import ForwardSubstitution


def test_iterative_forward_substitution_matches_independent_global_system():
    solver = ForwardSubstitution(space_hierarchy(1))
    s = solver.space
    diagonal = sparse.kron(solver.kt, s.mass) + sparse.kron(solver.mt, s.stiffness)
    coupling = sparse.kron(solver.nt, s.mass)
    a = sparse.kron(sparse.eye(8), diagonal) - sparse.kron(
        sparse.diags([np.ones(7)], [-1], shape=(8, 8)), coupling
    )
    rhs = np.random.default_rng(91).normal(size=(8, solver.q, solver.n))
    reference = spsolve(a.tocsc(), rhs.ravel()).reshape(rhs.shape)
    u, iterations, residuals = solver.solve(rhs, tol=1e-11)
    assert solver.apply(u).ravel() == pytest.approx(a @ u.ravel(), abs=1e-12)
    assert u == pytest.approx(reference, rel=1e-9, abs=1e-9)
    assert max(residuals) <= 1e-11
    assert min(iterations) > 0
    zero, counts, residuals = solver.solve(np.zeros_like(rhs))
    assert not zero.any() and max(counts) == max(residuals) == 0
    with pytest.raises(ArithmeticError, match="spatial solve failed"):
        solver.solve(rhs, maxiter=1, tol=1e-14)
