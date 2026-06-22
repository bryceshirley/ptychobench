# src/ptychobench/solver.py
import numpy as np
import logging
from typing import Callable, Type, Optional, Sequence
from dataclasses import dataclass

from ptychobench.grid import SimulationGrid
from ptychobench.operators import ForwardOperator
from ptychobench.metrics import calculate_rmse

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    histories: dict[str, np.ndarray]
    errors: dict[str, float]
    history_eps: np.ndarray
    sample_name: str
    sample_params: dict


def run_benchmark(
    grid: SimulationGrid,
    sample: Callable,
    operators: Sequence[Type[ForwardOperator]],
    sample_params: Optional[dict] = None,
) -> BenchmarkResult:
    """
    Runs the main z-propagation loop for all provided operators.
    """
    # 1. Initialization & Dependency Injection
    # Instantiate the classes with the grid and build the working dictionary
    instantiated_ops = {}
    for OpClass in operators:
        op_instance = OpClass(grid)
        instantiated_ops[op_instance.name] = op_instance

    psi_0 = grid.get_initial_field()

    current_psi = {name: psi_0.copy() for name in instantiated_ops.keys()}
    histories = {
        name: np.zeros((grid.Nz, grid.N), dtype=complex)
        for name in instantiated_ops.keys()
    }
    history_eps = np.zeros((grid.Nz, grid.N), dtype=complex)

    kwargs = sample_params or {}

    logger.debug(f"Starting 2D propagation over {grid.Nz} steps...")

    # 2. The Range-Dependent Z-Loop
    for i, z in enumerate(grid.z_steps):
        eps = sample(grid, z, **kwargs)
        E = np.diag(eps)
        history_eps[i, :] = eps

        # Propagate every operator
        for name, operator in instantiated_ops.items():
            histories[name][i, :] = current_psi[name]
            P = operator.step(E)
            current_psi[name] = P @ current_psi[name]

        if (i + 1) % max(1, (grid.Nz // 10)) == 0:
            logger.debug(f"Step {i + 1}/{grid.Nz} completed.")

    # 3. Compute Relative Errors against "Exact"
    errors = {}
    if "Exact" in histories:
        exact_data = histories["Exact"]
        for name, data in histories.items():
            if name == "Exact":
                errors[name] = 0.0
            else:
                errors[name] = calculate_rmse(exact_data.ravel(), data.ravel())
    else:
        logger.warning(
            "'Exact' key not found in operators. Skipping error calculation."
        )

    logger.debug("Propagation complete.")

    # Extract the function name dynamically (or default to 'unknown_sample')
    s_name = getattr(sample, "__name__", "unknown_sample")

    return BenchmarkResult(
        histories=histories,
        errors=errors,
        history_eps=history_eps,
        sample_name=s_name,
        sample_params=kwargs,
    )
