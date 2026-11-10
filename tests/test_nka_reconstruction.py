import numpy as np
import pytest
from actapint.nka_reproduction import operators, frobenius_weights


def test_closed_form_weights_match_full_matrix_normal_equations():
    for nu in (0.02, 0.002):
        a, d, x = operators(nu, 1 / 20)
        states = np.random.default_rng(8).normal(size=(11, len(x)))
        jac = d[None] + 2 * a[None] * states[:, None, :]
        avg = jac.mean(axis=0)
        explicit = np.einsum("tij,ij->t", jac, avg) / np.sum(avg * avg)
        w = frobenius_weights(states, nu, 1 / 20)
        assert w == pytest.approx(explicit, abs=2e-15)
        assert w.mean() == pytest.approx(1)
        assert np.einsum(
            "tij,ij->t", jac - w[:, None, None] * avg, avg
        ) == pytest.approx(np.zeros(11), abs=2e-11)


def test_burgers_jacobian_directional_derivative_and_periodic_mass():
    a, d, x = operators(0.02, 1 / 20)
    assert len(x) == 20 and x[-1] < 1
    rng = np.random.default_rng(8)
    u = rng.normal(size=20)
    v = rng.normal(size=20)

    def f(z):
        return a @ z**2 + d @ z

    eps = 1e-5
    finite = (f(u + eps * v) - f(u - eps * v)) / (2 * eps)
    assert finite == pytest.approx((d + 2 * a * u[None, :]) @ v, abs=3e-9)
    assert f(u).sum() == pytest.approx(0, abs=1e-12)
