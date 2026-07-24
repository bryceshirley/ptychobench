# src/ptychobench_3d/benchmark.py
import logging
import gc
import tracemalloc
from time import time
from typing import Type, Sequence, Union, List, Dict, Tuple, cast

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ptychobench_3d.grid import SimulationGrid

# from ptychobench.operators import ForwardOperator, ExactOperator
from ptychobench_3d.operators import ExactOperatorMF as ExactOperator
from ptychobench_3d.operators import ForwardOperator
from ptychobench_3d.metrics import (
    calculate_rmse,
    calculate_rmse_intensity,
    calculate_max_intensity_error,
)
from ptychobench_3d.results import BenchmarkResult

logger = logging.getLogger(__name__)


def _prepare_operators(
    grid: SimulationGrid,
    operators: Union[Sequence[Type[ForwardOperator]], Type[ForwardOperator]],
) -> List[ForwardOperator]:
    """Parses input operators, ensures ExactOperator is present, and instantiates them."""
    op_classes: List[Type[ForwardOperator]]
    if isinstance(operators, type):
        op_classes = cast(List[Type[ForwardOperator]], [operators])
    else:
        op_classes = list(operators)

    if ExactOperator not in op_classes:
        op_classes.append(ExactOperator)

    instantiated_ops = [OpClass(grid) for OpClass in op_classes]

    # Check for duplicate display names to avoid dictionary key overwrites
    op_names = [op.name for op in instantiated_ops]
    if len(set(op_names)) != len(op_names):
        logger.warning(
            "Duplicate operator names detected. Downstream dictionary keys may overwrite each other."
        )

    return instantiated_ops


def _run_propagation(
    grid: SimulationGrid,
    instantiated_ops: List[ForwardOperator],
    sample: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Runs the full z-propagation loop for all operators with Memory and Speed Profiling."""
    psi_0 = grid.get_initial_field()

    current_psi = {op.name: psi_0.copy() for op in instantiated_ops}

    # Dynamically size the history array based on psi_0.shape
    wavefield_history = {
        op.name: np.zeros((grid.Nz, *psi_0.shape), dtype=complex)
        for op in instantiated_ops
    }

    logger.debug(f"Starting propagation over {grid.Nz} steps...")

    for operator in instantiated_ops:
        op_name = operator.name
        logger.info(f"--- Starting full propagation for: {op_name} ---")

        # --- Clean up memory before profiling ---
        gc.collect()
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # --- Start Tracking ---
        tracemalloc.start()
        t0 = time()

        for i in range(grid.Nz):
            # Fetch the permittivity array for this step using [i] for N-D safety
            eps = sample[i, :, :]

            # Save to wavefield_history using [i] for N-D safety
            wavefield_history[op_name][i] = current_psi[op_name]

            # Propagate
            current_psi[op_name] = operator.step(eps, current_psi[op_name])

            # Progress logging
            if (i + 1) % max(1, (grid.Nz // 10)) == 0:
                logger.debug(f"[{op_name}] Step {i + 1}/{grid.Nz} completed.")

        # --- Stop Tracking ---
        t1 = time()

        # CPU Memory Profile
        current_ram, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_ram / (1024 * 1024)

        # Device Memory Profile
        vram_str = "0.00 MB (CPU Only)"
        if torch is not None:
            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                vram_str = f"{peak_vram_mb:.2f} MB (CUDA Peak)"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                current_vram_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                vram_str = f"~{current_vram_mb:.2f} MB (MPS Current)"

        logger.info(
            f"Operator '{op_name}' Summary:\n"
            f"  - Time:       {t1 - t0:.4f} seconds\n"
            f"  - Peak RAM:   {peak_ram_mb:.2f} MB\n"
            f"  - Peak VRAM:  {vram_str}"
        )

    return wavefield_history


def _compute_metrics(
    instantiated_ops: List[ForwardOperator],
    wavefield_history: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Calculates RMSE and Max Error metrics relative to the ExactOperator."""
    rmse_wavefield: Dict[str, float] = {}
    rmse_detector: Dict[str, float] = {}
    max_error_detector: Dict[str, float] = {}

    exact_ops = [op for op in instantiated_ops if isinstance(op, ExactOperator)]

    if not exact_ops:
        logger.warning(
            "'ExactOperator' not found in operators. Skipping error calculation."
        )
        return rmse_wavefield, rmse_detector, max_error_detector

    exact_op_name = exact_ops[0].name
    exact_data = wavefield_history[exact_op_name]

    for op in instantiated_ops:
        op_name = op.name

        if op_name == exact_op_name:
            rmse_wavefield[op_name] = 0.0
            rmse_detector[op_name] = 0.0
            max_error_detector[op_name] = 0.0
        else:
            rmse_wavefield[op_name] = calculate_rmse(
                exact_data.ravel(), wavefield_history[op_name].ravel()
            )
            # Use [-1] instead of [-1, :] for N-dimensional arrays
            rmse_detector[op_name] = calculate_rmse_intensity(
                exact_data[-1], wavefield_history[op_name][-1]
            )
            max_error_detector[op_name] = calculate_max_intensity_error(
                exact_data[-1], wavefield_history[op_name][-1]
            )

    return rmse_wavefield, rmse_detector, max_error_detector


def run_benchmark(
    grid: SimulationGrid,
    sample: np.ndarray,
    operators: Union[Sequence[Type[ForwardOperator]], Type[ForwardOperator]],
) -> BenchmarkResult:
    """
    Runs the main z-propagation loop for all provided operators and computes error metrics.
    """
    # 1. Initialization
    instantiated_ops = _prepare_operators(grid, operators)

    # 2. Execution
    wavefield_history = _run_propagation(grid, instantiated_ops, sample)

    # 3. Metrics Calculation
    rmse_wavefield, rmse_detector, max_error_detector = _compute_metrics(
        instantiated_ops, wavefield_history
    )

    logger.debug("Propagation complete.")

    # 4. Result Packaging
    return BenchmarkResult(
        grid=grid,
        wavefield_history=wavefield_history,
        rmse_wavefield=rmse_wavefield,
        rmse_detector=rmse_detector,
        max_error_detector=max_error_detector,
        sample_history=sample,
    )
