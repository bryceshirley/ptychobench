from ptychobench.metrics import calculate_rmse_intensity
import numpy as np
import logging
from typing import Type, Sequence, Union
from ptychobench.samples import (
    Sample,
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
)
from ptychobench.operators import (
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
)
from ptychobench.grid import SimulationGrid
from ptychobench.operators import ForwardOperator, ExactOperator
from ptychobench.results import BenchmarkResult
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def run_exit_wave_benchmark(
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
        op_instance = OpClass(grid)  # ty:ignore[call-non-callable]
        # Use class name for internal tracking
        instantiated_ops[OpClass.__name__] = (
            op_instance  # ty:ignore[unresolved-attribute]
        )

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
    # errors = {}

    exit_wave_errors = {}

    # Ensure ExactOperator was part of the run
    if "ExactOperator" in instantiated_ops:
        # Look up what string the ExactOperator is using for its display name
        exact_display_name = instantiated_ops["ExactOperator"].name
        exact_data = wavefield_history[exact_display_name]

        for class_name, op in instantiated_ops.items():
            if class_name == "ExactOperator":
                # errors[class_name] = 0.0
                exit_wave_errors[class_name] = 0.0
            else:
                # Calculate RMSE and store it under the simple class name
                exit_wave_errors[class_name] = calculate_rmse_intensity(
                    exact_data[-1, :].ravel(), wavefield_history[op.name][-1, :].ravel()
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
        errors=exit_wave_errors,
        sample_history=sample_history,
        sample_name=s_name,
        sample_params=s_params,
    )


# NEW_TESTS = True
NEW_TESTS = False

# moduli = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
moduli = [0.1, 0.5, 1.0, 1.5, 2.0]

# Each tuple is a test with the format:
# (divergence_angle, modulus, period, core_width, sample_idx)
# period and core_width may be None for samples that do not have them as variables
tests = [(0.1, modulus, None, None, 0) for modulus in moduli]
SAMPLES = [
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
]

FILE_1 = "scripts/script_data/exit_wave_errors_exact.npy"
FILE_2 = "scripts/script_data/exit_wave_errors_paraxial.npy"
FILE_3 = "scripts/script_data/exit_wave_errors_feit_fleck.npy"
FILE_4 = "scripts/script_data/exit_wave_errors_lin_duda.npy"
errors_exact = []
errors_paraxial = []
errors_feit_fleck = []
errors_lin_duda = []


def do_tests():
    for test in tests:
        divergence_angle = test[0]
        modulus = test[1]
        period = test[2]
        core_width = test[3]
        sample = SAMPLES[test[4]]

        grid = SimulationGrid(divergence_angle=divergence_angle)

        if period is not None and core_width is not None:
            sample = sample(
                modulus=modulus,
                period=period,  # ty:ignore[unknown-argument]
                core_width=core_width,  # ty:ignore[unknown-argument]
            )
        elif period is not None:
            sample = sample(
                modulus=modulus,
                period=period,  # ty:ignore[unknown-argument]
            )
        elif core_width is not None:
            sample = sample(
                modulus=modulus,
                core_width=core_width,  # ty:ignore[unknown-argument]
            )
        else:
            sample = sample(modulus=modulus)
        operators = [ParaxialOperator, FeitFleckOperator, LinDudaOperator]

        result = run_exit_wave_benchmark(grid, sample, operators)
        errors_exact.append(result.errors["ExactOperator"])
        errors_paraxial.append(result.errors["ParaxialOperator"])
        errors_feit_fleck.append(result.errors["FeitFleckOperator"])
        errors_lin_duda.append(result.errors["LinDudaOperator"])


if NEW_TESTS:
    do_tests()
    np.save(arr=errors_exact, file=FILE_1)
    np.save(arr=errors_paraxial, file=FILE_2)
    np.save(arr=errors_feit_fleck, file=FILE_3)
    np.save(arr=errors_lin_duda, file=FILE_4)
else:
    try:
        errors_exact = np.load(FILE_1)
        errors_paraxial = np.load(FILE_2)
        errors_feit_fleck = np.load(FILE_3)
        errors_lin_duda = np.load(FILE_4)
    except OSError:
        do_tests()

fig, axes = plt.subplots()
plt.plot(moduli, errors_paraxial, "-o", c="red", label="Paraxial Operator")
plt.plot(moduli, errors_feit_fleck, "-o", c="yellow", label="Feit/Fleck Operator")
plt.plot(moduli, errors_lin_duda, "-o", c="green", label="Lin/Duda Operator")
axes.set_xlabel("Modulus")
axes.set_ylabel("RMSE")
plt.legend()
plt.show()
