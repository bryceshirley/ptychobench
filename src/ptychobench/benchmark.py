# src/ptychobench/solver.py
import numpy as np
import logging
from typing import Type, Sequence, Union

from ptychobench.grid import SimulationGrid
from ptychobench.operators import ForwardOperator, ExactOperator
from ptychobench.metrics import calculate_rmse
from ptychobench.samples import Sample
from ptychobench.results import BenchmarkResult

logger = logging.getLogger(__name__)


def run_benchmark(
    grid: SimulationGrid,
    sample: Sample,
    operators: Union[Sequence[Type[ForwardOperator]], Type[ForwardOperator]],
) -> BenchmarkResult:
    """
    Runs the main z-propagation loop for all provided operators.
    """
    logger.info("")
    logger.info(f"Running benchmark for sample: {sample.__class__.__name__}")

    # --- 1. Initialization & Dependency Injection ---
    if isinstance(operators, (list, tuple, set)):
        op_classes = list(operators)
    else:
        op_classes = [operators]

    if ExactOperator not in op_classes:
        op_classes.append(ExactOperator)

    instantiated_ops = {}
    for OpClass in op_classes:
        op_instance = OpClass(grid)
        # Use class name for internal tracking
        instantiated_ops[OpClass.__name__] = op_instance

    psi_0 = grid.get_initial_field()
    current_psi = {name: psi_0.copy() for name in instantiated_ops.keys()}

    # Key the wavefield_history dictionary with the beautiful display name (operator.name)
    wavefield_history = {
        op.name: np.zeros((grid.Nz, grid.N), dtype=complex)
        for op in instantiated_ops.values()
    }
    sample_history = np.zeros((grid.Nz, grid.N), dtype=complex)

    logger.debug(f"Starting 2D propagation over {grid.Nz} steps...")

    # --- 2. The Range-Dependent Z-Loop ---
    for i, z in enumerate(grid.z_steps):
        eps = sample.get_permittivity(grid, z)
        E = np.diag(eps)
        sample_history[i, :] = eps

        for class_name, operator in instantiated_ops.items():
            display_name = operator.name

            # Save to wavefield_history using the display name
            wavefield_history[display_name][i, :] = current_psi[class_name]

            # Propagate using the internal class name
            P = operator.step(E)
            current_psi[class_name] = P @ current_psi[class_name]

        if (i + 1) % max(1, (grid.Nz // 10)) == 0:
            logger.debug(f"Step {i + 1}/{grid.Nz} completed.")

    # --- 3. Compute Relative Errors against "ExactOperator" ---
    errors = {}

    # Ensure ExactOperator was part of the run
    if "ExactOperator" in instantiated_ops:
        # Look up what string the ExactOperator is using for its display name
        exact_display_name = instantiated_ops["ExactOperator"].name
        exact_data = wavefield_history[exact_display_name]

        for class_name, op in instantiated_ops.items():
            if class_name == "ExactOperator":
                errors[class_name] = 0.0
            else:
                # Calculate RMSE and store it under the simple class name
                errors[class_name] = calculate_rmse(
                    exact_data.ravel(), wavefield_history[op.name].ravel()
                )
    else:
        logger.warning(
            "'ExactOperator' not found in operators. Skipping error calculation."
        )

    logger.debug("Propagation complete.")

    s_name = sample.__class__.__name__
    s_params = vars(sample).copy()

    return BenchmarkResult(
        grid=grid,
        wavefield_history=wavefield_history,
        errors=errors,
        sample_history=sample_history,
        sample_name=s_name,
        sample_params=s_params,
    )
