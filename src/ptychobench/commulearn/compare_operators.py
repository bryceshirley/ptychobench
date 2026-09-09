"""
Issue #56: compare the trained FNO against the classical operators.

    1. Reproduce the benchmark plots, with the FNO alongside the exact solution
       and the classical operators.
    2. Plot the error between the ground truth and each operator.

The FNO joins the run as one more entry in the result's wavefield history, so the
figures come from :mod:`ptychobench.plotting` and nothing is drawn here.

Run generate_data.py and then train_fno.py first: this script plots the model
that train_fno.py trained and saved, it does not train one of its own.
"""

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ptychobench.samples as samples_module
from ptychobench.benchmark import run_benchmark
from ptychobench.commulearn.dataset import DATA_PATH
from ptychobench.commulearn.train_fno import CONFIG as TRAINING_CONFIG
from ptychobench.commulearn.train_fno import TrainingConfig, build_model
from ptychobench.grid import SimulationGrid
from ptychobench.numerics.metrics import (
    calculate_farfield_wave,
    calculate_max_error,
    calculate_rmse,
)
from ptychobench.operators import FeitFleckOperator, ForwardOperator
from ptychobench.plotting import (
    plot_evolution_1D,
    plot_farfield_error,
    plot_sample,
)
from ptychobench.results import BenchmarkResult
from ptychobench.samples import Sample

SAVE_DIR = Path("scripts/results")

#: The key the FNO's trajectory is filed under, alongside the operator class names.
FNO_NAME = "FNOOperator"

CONFIG = 0  # Which of the num_configs configurations to plot
Z_STEP = 0  # Which z-step of that configuration to label as trained or unseen


def find_rows(data: Mapping[str, Any], config: int, z_step: int) -> list[int]:
    """
    Return the row index of every sample type at one configuration and z-step.

    Reads the labels generate_data.py stored, so it does not depend on the order
    the generation loops happened to run in.
    """
    return [
        row
        for row in range(len(data["sample_names"]))
        if data["config_ids"][row] == config and data["z_indices"][row] == z_step
    ]


def rebuild_sample(data: Mapping[str, Any], row: int) -> Sample:
    """
    Rebuild the sample a row was generated from, so run_benchmark can rerun it.

    generate_data.py records the class name and the constructor arguments, which
    is everything the sample needs to come back identical.
    """
    sample_class = getattr(samples_module, data["sample_names"][row])
    return sample_class(**data["sample_params"][row])


def load_trained_model(config: TrainingConfig = TRAINING_CONFIG):
    """Load the model train_fno.py trained, so both scripts report the same one."""
    if not config.weights_path.exists():
        raise FileNotFoundError(
            f"No weights at {config.weights_path}. Run train_fno.py first."
        )

    model = build_model(config)
    # The FNO's state_dict stores neuralop objects, too many to allowlist one by
    # one, so fall back to full pickle. Safe here: train_fno.py wrote this file
    model.load_state_dict(torch.load(config.weights_path, weights_only=False))
    print(f"Loaded the model trained by train_fno.py from {config.weights_path}")
    return model


def fno_step(model, psi: Any, eps: Any) -> np.ndarray:
    """
    Advance the field by one step with the FNO.

    The counterpart of :meth:`~ptychobench.operators.ForwardOperator.step`, but
    taking the field and the permittivity stacked as real channels rather than an
    environment matrix, because that is what the network was trained on.
    """
    psi = np.asarray(torch.as_tensor(psi).cpu())
    eps = np.asarray(torch.as_tensor(eps).cpu())

    stacked = torch.stack(
        [
            torch.from_numpy(np.real(psi)).float(),
            torch.from_numpy(np.imag(psi)).float(),
            torch.from_numpy(np.real(eps)).float(),
        ]
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction = model(stacked)[0]

    return (prediction[0] + 1j * prediction[1]).numpy().astype(complex)


def fno_history(model, grid: SimulationGrid, sample: Sample) -> np.ndarray:
    """
    Propagate the FNO through the whole sample, one step at a time.

    Chaining single-step predictions is what makes the FNO comparable with the
    classical operators, which run_benchmark applies the same way. It also lets
    any error compound along z, as it does for them.

    Returns
    -------
    numpy.ndarray
        The ``(Nz, N)`` trajectory, recorded at the start of each step to match
        the convention run_benchmark records the classical operators with.
    """
    psi = np.asarray(torch.as_tensor(grid.get_initial_field()).cpu())
    history = np.empty((grid.Nz, grid.N), dtype=complex)

    for i, z in enumerate(grid.z_steps):
        history[i] = psi
        psi = fno_step(model, psi, sample.get_permittivity(grid, z))

    return history


def _measure(histories: Mapping[str, np.ndarray]) -> tuple[dict, dict, dict]:
    """
    Measure every trajectory against the reference, the way run_benchmark does.

    Recomputed rather than reused, because run_benchmark worked its metrics out
    before the FNO joined the result.
    """
    reference = histories[BenchmarkResult.REFERENCE]
    reference_intensity = calculate_farfield_wave(reference[-1, :], mode="intensity")

    rmse_wavefield, rmse_detector, max_error_detector = {}, {}, {}
    for name, history in histories.items():
        intensity = calculate_farfield_wave(history[-1, :], mode="intensity")
        rmse_wavefield[name] = calculate_rmse(reference.ravel(), history.ravel())
        rmse_detector[name] = calculate_rmse(reference_intensity, intensity)
        max_error_detector[name] = calculate_max_error(reference_intensity, intensity)

    return rmse_wavefield, rmse_detector, max_error_detector


def benchmark_with_fno(
    grid: SimulationGrid,
    sample: Sample,
    model,
    operators: Sequence[type[ForwardOperator]] = (FeitFleckOperator,),
) -> BenchmarkResult:
    """
    Run the classical benchmark, then add the FNO as one more operator.

    Adding its trajectory to the result is what lets every figure come from
    ptychobench.plotting: those functions walk the wavefield history, so the FNO
    is drawn beside the classical operators without a line of plotting here.
    """
    result = run_benchmark(grid, sample, list(operators))

    result.wavefield_history[FNO_NAME] = fno_history(model, grid, sample)
    result.display_names[FNO_NAME] = "FNO"

    rmse_wavefield, rmse_detector, max_error_detector = _measure(
        result.wavefield_history
    )
    result.rmse_wavefield = rmse_wavefield
    result.rmse_detector = rmse_detector
    result.max_error_detector = max_error_detector
    return result


def compare(
    path: Path = DATA_PATH,
    config: int = CONFIG,
    z_step: int = Z_STEP,
    save_dir: Path = SAVE_DIR,
) -> None:
    """
    Compare the FNO against the classical operators, one figure set per sample.

    Parameters
    ----------
    path : pathlib.Path, optional
        The dataset to read the sample parameters back from.
    config : int, optional
        Which draw of random parameters to plot.
    z_step : int, optional
        Which z-step decides whether a sample is labelled trained or unseen.
    save_dir : pathlib.Path, optional
        Where the figures are written.
    """
    data = torch.load(path, weights_only=False)
    grid = SimulationGrid(**data["grid_params"])
    model = load_trained_model()

    # Reproduce the split train_fno.py used, so each sample can be labelled
    generator = torch.Generator().manual_seed(TRAINING_CONFIG.split_seed)
    trained_rows = set(
        torch.randperm(len(data["sample_names"]), generator=generator)[
            : TRAINING_CONFIG.batch_size
        ].tolist()
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'sample':24}  {'seen?':8}  {'Feit/Fleck':>12}  {'FNO':>12}")

    for row in find_rows(data, config, z_step):
        sample = rebuild_sample(data, row)
        name = data["sample_names"][row]
        seen = "trained" if row in trained_rows else "unseen"

        result = benchmark_with_fno(grid, sample, model)
        print(
            f"{name:24}  {seen:8}  "
            f"{result.rmse_wavefield['FeitFleckOperator']:12.4e}  "
            f"{result.rmse_wavefield[FNO_NAME]:12.4e}"
        )

        # 1. The wavefields, with the FNO drawn alongside the classical operators
        phase, amplitude = plot_evolution_1D(result)
        # 2. The error of each operator against the ground truth
        error = plot_farfield_error(result, log_scale=True)
        sample_figure = plot_sample(result)

        for label, figure in [
            ("evolution_phase", phase),
            ("evolution_amplitude", amplitude),
            ("farfield_error", error),
            ("sample", sample_figure),
        ]:
            figure.suptitle(f"{name} ({seen})")
            figure.savefig(save_dir / f"{name}_{label}.png", dpi=150)
            figure.clf()

    print(f"\nSaved the figures to {save_dir}")


if __name__ == "__main__":
    # Optionally pick the configuration and z-step: compare_operators.py CONFIG Z_STEP
    if len(sys.argv) == 3:
        compare(config=int(sys.argv[1]), z_step=int(sys.argv[2]))
    else:
        compare()
