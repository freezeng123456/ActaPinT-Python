import numpy as np
import pytest

from actapint.gpu import (
    CupyBurgersFineStepper,
    cupy_available,
    resolve_backend,
)
from actapint.integrators import burgers_fine_step
from actapint.operators import burgers_operators


def test_cpu_backend_is_always_available():
    assert resolve_backend("cpu") == "cpu"


@pytest.mark.skipif(not cupy_available(), reason="CUDA-capable CuPy is unavailable")
def test_batched_gpu_burgers_matches_cpu():
    n, batch, substeps = 12, 3, 2
    dt = 0.0025
    a1, a2 = burgers_operators(n, 1 / n, 0.1, periodic=True)
    values = np.random.default_rng(7).random((n, batch))

    gpu_step = CupyBurgersFineStepper(a1, a2, dt, substeps)
    actual = gpu_step(values)
    expected = np.column_stack(
        [
            burgers_fine_step(
                a1,
                a2,
                values[:, j],
                dt,
                substeps,
                "sdirk22",
                1e-12,
            )
            for j in range(batch)
        ]
    )

    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)
