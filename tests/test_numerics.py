import numpy as np

from actapint.integrators import linear_step_matrix
from actapint.operators import (
    periodic_first_derivative,
    periodic_negative_laplacian,
)


def test_periodic_operators_annihilate_constants():
    n = 32
    one = np.ones(n)
    assert np.linalg.norm(periodic_first_derivative(n, 1 / n) @ one, np.inf) < 1e-13
    assert np.linalg.norm(periodic_negative_laplacian(n, 1 / n) @ one, np.inf) < 1e-11


def test_periodic_negative_laplacian_is_positive_semidefinite():
    matrix = periodic_negative_laplacian(24, 1 / 24).toarray()
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > -1e-10


def test_backward_euler_damps_diffusion():
    operator = periodic_negative_laplacian(32, 1 / 32, scale=0.01)
    propagator = linear_step_matrix(operator, 0.1, "ie")
    eigenvalues = np.linalg.eigvals(propagator)
    assert np.max(np.abs(eigenvalues)) <= 1 + 1e-12


def test_trapezoidal_preserves_wave_like_imaginary_mode():
    operator = np.array([[0.0, -2.0], [2.0, 0.0]])
    propagator = linear_step_matrix(operator, 0.05, "tr")
    assert np.allclose(propagator.T @ propagator, np.eye(2), atol=1e-12)
