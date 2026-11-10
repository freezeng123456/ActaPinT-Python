import numpy as np
import pytest
from scipy.linalg import eigh
from scipy import sparse
from scipy.sparse.linalg import spsolve
from actapint.stmg_fem import (
    dg_matrices,
    space_hierarchy,
    TimeMultigrid,
    cube_mesh,
    refine_mesh,
)


def test_degree_zero_is_backward_euler():
    k, m, n, p = dg_matrices(0, 0.1)
    assert k == pytest.approx(np.array([[1.0]]))
    assert m == pytest.approx(np.array([[0.1]]))
    assert n == pytest.approx(np.array([[1.0]]))
    assert p == pytest.approx(np.array([[1.0], [1.0]]))


def test_dg_galerkin_coarsening():
    for degree in (0, 1, 3):
        k, m, n, p = dg_matrices(degree, 0.1)
        kc, mc, nc, _ = dg_matrices(degree, 0.2)
        fine = np.block([[k, np.zeros_like(k)], [-n, k]])
        assert p.T @ fine @ p == pytest.approx(kc, abs=2e-12)
        assert p.T @ np.kron(np.eye(2), m) @ p == pytest.approx(mc, abs=2e-12)
        assert k @ np.ones(degree + 1) == pytest.approx(
            n @ np.ones(degree + 1), abs=1e-12
        )


def test_tetrahedral_refinement_preserves_volume_and_galerkin_operators():
    points, tets = cube_mesh()
    for _ in range(3):
        vol = np.abs(np.linalg.det(points[tets[:, 1:]] - points[tets[:, 0, None]])) / 6
        assert vol.min() > 0
        assert vol.sum() == pytest.approx(1.0)
        points, tets, _ = refine_mesh(points, tets)
    levels = space_hierarchy(2)
    assert [len(l.nodes) for l in levels] == [1, 9, 91]
    for coarse, fine in zip(levels, levels[1:]):
        p = fine.prolong
        assert (p.T @ fine.mass @ p).toarray() == pytest.approx(
            coarse.mass.toarray(), abs=1e-13
        )
        assert (p.T @ fine.stiffness @ p).toarray() == pytest.approx(
            coarse.stiffness.toarray(), abs=1e-12
        )


@pytest.mark.parametrize("spatial_solver", ["direct", "vcycle"])
def test_multigrid_matches_independent_global_solve(spatial_solver):
    levels = space_hierarchy(1)
    mg = TimeMultigrid(levels, 8, spatial_solver=spatial_solver)
    sp = levels[-1]
    block = sparse.kron(mg.kt, sp.mass) + sparse.kron(mg.mt, sp.stiffness)
    coupling = sparse.kron(mg.nt, sp.mass)
    global_a = sparse.kron(sparse.eye(8), block) - sparse.kron(
        sparse.diags([np.ones(7)], [-1], shape=(8, 8)), coupling
    )
    rhs = np.random.default_rng(4).normal(size=(8, 4, mg.n))
    reference = spsolve(global_a.tocsc(), rhs.ravel()).reshape(rhs.shape)
    u, history = mg.solve(rhs, tol=1e-11)
    assert history[-1] < 1e-11
    assert len(history) < 30
    assert u == pytest.approx(reference, rel=1e-8, abs=1e-9)
    assert mg.sequential_solve(rhs) == pytest.approx(reference, rel=1e-11, abs=1e-11)


def test_dg3_resolves_semidiscrete_heat_eigenmode():
    levels = space_hierarchy(1)
    sp = levels[-1]
    eigenvalues, vectors = eigh(sp.stiffness.toarray(), sp.mass.toarray())
    phi = vectors[:, 0]
    lam = eigenvalues[0]
    errors = []
    for steps in (4, 8):
        dt = 0.1 / steps
        mg = TimeMultigrid(levels, steps, dt=dt)
        rhs = np.zeros((steps, 4, mg.n))
        rhs[0] = mg.nt @ np.tile(sp.mass @ phi, (4, 1))
        sol = mg.sequential_solve(rhs)
        from numpy.polynomial import Polynomial
        from numpy.polynomial.legendre import leggauss

        nodes = (leggauss(4)[0] + 1) / 2
        right = np.array(
            [
                (lambda p: p(1) / p(nodes[j]))(
                    Polynomial.fromroots(np.delete(nodes, j))
                )
                for j in range(4)
            ]
        )
        end = right @ sol[-1]
        errors.append(np.linalg.norm(end - np.exp(-lam * 0.1) * phi))
    assert errors[1] < errors[0] / 80
    assert errors[1] < 1e-7
