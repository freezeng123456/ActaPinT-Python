"""Python reproduction of the ActaPinT MATLAB experiments."""

from .experiments import (
    run_linear_solution,
    run_burgers_solution,
    run_wave_solution,
    run_parareal_linear,
    run_parareal_burgers,
    run_mgrit_linear,
    run_iterative_paradiag_ade,
    run_stmg_linear,
)

__all__ = [
    "run_linear_solution",
    "run_burgers_solution",
    "run_wave_solution",
    "run_parareal_linear",
    "run_parareal_burgers",
    "run_mgrit_linear",
    "run_iterative_paradiag_ade",
    "run_stmg_linear",
]

