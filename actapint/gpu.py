"""Optional GPU kernels for independent time-slice propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU backend requested, but CuPy is not installed. Install the CuPy "
            "package matching the server CUDA runtime (for CUDA 12: "
            "`python -m pip install cupy-cuda12x`)."
        ) from exc
    return cp


def cupy_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""
    try:
        cp = _cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    """Resolve ``auto`` to CUDA when available and otherwise to CPU."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unknown compute backend: {requested}")
    if requested == "cpu":
        return "cpu"
    available = cupy_available()
    if requested == "gpu" and not available:
        _cupy()
        raise RuntimeError("GPU backend requested, but no CUDA device is available.")
    return "gpu" if available else "cpu"


def backend_metadata(backend: str) -> dict[str, object]:
    """Return JSON-serializable backend and device metadata."""
    if backend == "cpu":
        return {"backend": "cpu"}
    cp = _cupy()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "backend": "gpu",
        "library": f"CuPy {cp.__version__}",
        "device_index": int(device.id),
        "device_name": str(name),
        "compute_capability": (
            f"{int(properties['major'])}.{int(properties['minor'])}"
        ),
        "memory_bytes": int(properties["totalGlobalMem"]),
    }


@dataclass
class CupyBurgersFineStepper:
    """Batched SDIRK22 propagation for independent Burgers time slices.

    The spatial operators are small in the paper experiments, but Parareal
    supplies many independent right-hand sides. CuPy's batched dense solver
    exposes that time-slice parallelism to cuSOLVER while keeping all Newton
    systems for one batch on the device.
    """

    a1: object
    a2: object
    dt: float
    substeps: int
    tol: float = 1e-12
    max_iter: int = 80

    def __post_init__(self) -> None:
        cp = _cupy()
        self._cp = cp
        self._a1 = cp.asarray(self.a1.toarray(), dtype=cp.float64)
        self._a2 = cp.asarray(self.a2.toarray(), dtype=cp.float64)
        self._n = self._a1.shape[0]
        self._gamma = (2.0 - np.sqrt(2.0)) / 2.0
        self._tau = self._gamma * self.dt
        self._base_jacobian = (
            cp.eye(self._n, dtype=cp.float64) + self._tau * self._a2
        )

    def _implicit_euler(self, rhs):
        cp = self._cp
        z = rhs.copy()
        for _ in range(self.max_iter):
            residual = z + self._tau * (
                self._a2 @ z + self._a1 @ (z * z)
            ) - rhs
            if float(cp.max(cp.abs(residual)).get()) <= self.tol:
                return z
            jacobian = self._base_jacobian[None, :, :] + (
                2.0
                * self._tau
                * self._a1[None, :, :]
                * z.T[:, None, :]
            )
            correction = cp.linalg.solve(
                jacobian,
                residual.T[..., None],
            )[..., 0].T
            z -= correction
        raise RuntimeError("batched GPU Newton iteration did not converge")

    def __call__(self, values) -> np.ndarray:
        """Propagate an ``(space, time-slice)`` batch and return a NumPy array."""
        cp = self._cp
        values = np.asarray(values, dtype=float)
        squeeze = values.ndim == 1
        if squeeze:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self._n:
            raise ValueError(
                f"expected ({self._n}, batch) values, received {values.shape}"
            )

        z = cp.asarray(values)
        for _ in range(self.substeps):
            g1 = self._implicit_euler(z)
            rhs = z - (1.0 - self._gamma) * self.dt * (
                self._a2 @ g1 + self._a1 @ (g1 * g1)
            )
            z = self._implicit_euler(rhs)
        cp.cuda.Stream.null.synchronize()
        result = cp.asnumpy(z)
        return result[:, 0] if squeeze else result
