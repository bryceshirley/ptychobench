"""
The propagation loop: run every operator over the same sample and compare.

:func:`run_benchmark` is the entry point and the only public name here. It is
deliberately thin -- it resolves the operator list, drives the z-loop, and hands
the trajectory to the metrics -- with each of those steps behind a helper below
so the loop itself reads as the sequence it is.

The helpers are private because they are a decomposition of one function rather
than an API: their signatures are free to change, and a caller wanting to drive
the propagation directly should write the four-line loop against
:meth:`~ptychobench.operators.ForwardOperator.step` instead. See the technical
notes, "Which integrators carry gradients", for why a training loop in
particular should not go through here.
"""

import logging
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Sequence, Type, Union

import numpy as np

from ptychobench.grid import SimulationGrid
from ptychobench.numerics.metrics import (
    calculate_farfield_wave,
    calculate_max_error,
    calculate_rmse,
)
from ptychobench.operators import GroundTruthOperator, ForwardOperator, Integrator
from ptychobench.results import BenchmarkResult
from ptychobench.samples import Sample
from ptychobench.utils import to_numpy

logger = logging.getLogger(__name__)

#: How many steps of trajectory to buffer on the device before copying a whole
#: block to the host. A device-to-host copy is the expensive part of recording
#: a trajectory on a GPU backend -- each one drains the command queue, so
#: copying per step turns an Nz-step run into Nz synchronisation points, which
#: on Apple MPS roughly doubles the cost of a small-N run. Buffering costs
#: device memory for the block, which is why this is bounded rather than "the
#: whole run": a long trajectory would otherwise have to fit on the GPU twice.
_FLUSH_EVERY = 64


def _resolve_operators(
    grid: SimulationGrid,
    operators: Union[Sequence[Type[ForwardOperator]], Type[ForwardOperator]],
    integrator: Integrator,
) -> dict[str, ForwardOperator]:
    """
    Instantiate the operators for a run, keyed by class name.

    Parameters
    ----------
    grid : SimulationGrid
        The grid every operator is constructed against.
    operators : type or sequence of type
        The operator classes to run. ``GroundTruthOperator`` is appended if
        absent, since every error is measured against it.
    integrator : Integrator
        Applied to each operator that supports it; an operator with no
        split-step form falls back to ``"krylov"`` and the fallback is logged.

    Returns
    -------
    dict of str to ForwardOperator
        The instances, keyed by class name.

    Notes
    -----
    Keyed by class name rather than display name: the two used to be mixed, and
    two operators sharing a label then shared a history entry, with both RMSEs
    computed against whichever wrote to it last.
    """
    # Identify the single-class case by what it *is* rather than by ruling out
    # list/tuple/set, so any Sequence is accepted and the two union members are
    # distinguished unambiguously.
    if isinstance(operators, type) and issubclass(operators, ForwardOperator):
        op_classes: list[Type[ForwardOperator]] = [operators]
    else:
        op_classes = list(operators)

    if GroundTruthOperator not in op_classes:
        op_classes.append(GroundTruthOperator)

    resolved: dict[str, ForwardOperator] = {}
    for OpClass in op_classes:
        op_integrator = integrator
        if integrator == "split-step" and not OpClass.supports_split_step():
            op_integrator = "krylov"
            logger.info(
                f"{OpClass.__name__} has no split-step form; evaluating it with "
                "'krylov' instead of 'split-step'."
            )
        resolved[OpClass.__name__] = OpClass(grid, integrator=op_integrator)

    return resolved


class _TrajectoryRecorder:
    """
    Accumulate the trajectory on the device and copy it across in blocks.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the namespace and the trajectory shape ``(Nz, N)``.
    names : sequence of str
        One history is allocated per name.

    Notes
    -----
    The recorded values do not depend on the block size; see
    :data:`_FLUSH_EVERY` for why there is one. Splitting this out of the loop is
    what lets the block boundary be tested directly -- a stitching bug misplaces
    whole rows rather than perturbing them, which no convergence test notices.

    The histories are host NumPy ``complex128`` whatever backend the run used,
    so a benchmark number does not silently change precision because the run
    landed on a GPU.
    """

    def __init__(self, grid: SimulationGrid, names: Sequence[str]):
        self.xp = grid.xp
        self.sample_history = np.zeros((grid.Nz, grid.N), dtype=complex)
        self.wavefield_history = {
            name: np.zeros((grid.Nz, grid.N), dtype=complex) for name in names
        }
        self._sample_block: list = []
        self._wavefield_blocks: dict[str, list] = {name: [] for name in names}
        self._flushed = 0  # steps already written into the host arrays

    def record(self, eps: Any, fields: Mapping[str, Any]) -> None:
        """
        Append one step's permittivity and fields, flushing when full.

        Parameters
        ----------
        eps : Any
            The permittivity at this step.
        fields : mapping of str to Any
            The state *before* the step, which is the convention the plotting
            code reads: row ``i`` of a history is the field at ``z_steps[i]``,
            not after it.
        """
        self._sample_block.append(eps)
        for name, block in self._wavefield_blocks.items():
            block.append(fields[name])

        if len(self._sample_block) == _FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        """Copy the buffered block to the host. A no-op if nothing is buffered."""
        if not self._sample_block:
            return

        stop = self._flushed + len(self._sample_block)
        self.sample_history[self._flushed : stop, :] = to_numpy(
            self.xp.stack(self._sample_block)
        )
        self._sample_block.clear()

        for name, block in self._wavefield_blocks.items():
            self.wavefield_history[name][self._flushed : stop, :] = to_numpy(
                self.xp.stack(block)
            )
            block.clear()

        self._flushed = stop


def _error_metrics(
    wavefield_history: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Measure every operator's error against the reference.

    Parameters
    ----------
    wavefield_history : mapping of str to numpy.ndarray
        One ``(Nz, N)`` trajectory per operator, keyed by class name.

    Returns
    -------
    rmse_wavefield : dict of str to float
        RMSE over the entire trajectory.
    rmse_detector : dict of str to float
        RMSE of the far-field intensity at the final plane.
    max_error_detector : dict of str to float
        Maximum absolute far-field intensity error at the final plane.
    """
    rmse_wavefield: dict[str, float] = {}
    rmse_detector: dict[str, float] = {}
    max_error_detector: dict[str, float] = {}

    if BenchmarkResult.REFERENCE not in wavefield_history:
        # Unreachable through run_benchmark, which appends the reference -- but
        # this function does not know that, and returning three empty
        # dictionaries beats a KeyError from the loop below.
        logger.warning(
            f"{BenchmarkResult.REFERENCE!r} not found in operators. "
            "Skipping error calculation."
        )
        return rmse_wavefield, rmse_detector, max_error_detector

    reference = wavefield_history[BenchmarkResult.REFERENCE]
    reference_intensity = calculate_farfield_wave(reference[-1, :], mode="intensity")
    for class_name, data in wavefield_history.items():
        if class_name == BenchmarkResult.REFERENCE:
            rmse_wavefield[class_name] = 0.0
            rmse_detector[class_name] = 0.0
            max_error_detector[class_name] = 0.0
        else:
            intensity = calculate_farfield_wave(data[-1, :], mode="intensity")
            rmse_wavefield[class_name] = calculate_rmse(reference.ravel(), data.ravel())
            rmse_detector[class_name] = calculate_rmse(reference_intensity, intensity)
            max_error_detector[class_name] = calculate_max_error(
                reference_intensity, intensity
            )

    return rmse_wavefield, rmse_detector, max_error_detector


def _sample_parameters(sample: Sample) -> dict[str, Any]:
    """
    Extract the sample's configuration, for the record written into the result.

    Parameters
    ----------
    sample : Sample
        The sample the run used.

    Returns
    -------
    dict
        The declared dataclass fields, or ``vars()`` for a plain class.

    Notes
    -----
    Declared fields beat ``__dict__``: ClassVar caches are skipped and
    ``__slots__`` would not break it. Read shallowly rather than with
    ``asdict()``, which deep-copies and would duplicate any cached array.
    """
    if is_dataclass(sample):
        return {f.name: getattr(sample, f.name) for f in fields(sample)}
    return vars(sample).copy()


def run_benchmark(
    grid: SimulationGrid,
    sample: Sample,
    operators: Union[Sequence[Type[ForwardOperator]], Type[ForwardOperator]],
    integrator: Integrator = "direct",
) -> BenchmarkResult:
    """
    Run the main z-propagation loop for all provided operators.

    Parameters
    ----------
    grid : SimulationGrid
        The simulation grid.
    sample : Sample
        Supplies the permittivity at each z.
    operators : type or sequence of type
        The operator classes to compare.
        :class:`~ptychobench.operators.GroundTruthOperator` is appended if
        absent, since it supplies the reference.
    integrator : {"direct", "krylov", "split-step"}, optional
        How every operator evaluates its exponential. Applied uniformly where
        it can be: the benchmark exists to measure the difference between the
        operator *approximations*, and using one evaluator throughout keeps its
        error largely common to both sides rather than folding a second,
        unrelated error into every RMSE. Defaults to ``"direct"``.

        ``"split-step"`` is the exception, because it needs a generator built
        from Fourier-diagonal and real-space-diagonal pieces. All four
        approximations have one; the reference does not and never can, so it
        falls back to ``"krylov"`` and the fallback is logged. The two sides of
        every comparison in such a run are then evaluated differently, so a
        split-step run measures the splitting error *on top of* the operator
        error rather than in isolation; use ``"direct"`` or ``"krylov"`` to
        separate the two.

    Returns
    -------
    BenchmarkResult
        Wavefield histories and the error metrics against the reference.
    """
    logger.info("")
    logger.info(f"Running benchmark for sample: {sample.__class__.__name__}")

    operator_instances = _resolve_operators(grid, operators, integrator)
    recorder = _TrajectoryRecorder(grid, list(operator_instances))

    # Each operator gets its own array so that a future in-place `step` could
    # not make one operator's trajectory depend on another's. `.copy()` is a
    # NumPy method, not an Array API one, and torch does not have it.
    psi_0 = grid.get_initial_field()
    current_psi = {
        name: grid.xp.asarray(psi_0, copy=True) for name in operator_instances
    }

    logger.debug(f"Starting 2D propagation over {grid.Nz} steps...")
    for i, z in enumerate(grid.z_steps):
        eps = sample.get_permittivity(grid, z)
        recorder.record(eps, current_psi)

        for name, operator in operator_instances.items():
            # The permittivity goes in as a vector: the dense path builds
            # diag(eps) itself, and the matrix-free paths would only have to
            # undo it.
            current_psi[name] = operator.step(eps, current_psi[name])

        if (i + 1) % max(1, (grid.Nz // 10)) == 0:
            logger.debug(f"Step {i + 1}/{grid.Nz} completed.")

    recorder.flush()
    logger.debug("Propagation complete.")

    rmse_wavefield, rmse_detector, max_error_detector = _error_metrics(
        recorder.wavefield_history
    )

    return BenchmarkResult(
        grid=grid,
        display_names={
            name: op.display_name for name, op in operator_instances.items()
        },
        wavefield_history=recorder.wavefield_history,
        rmse_wavefield=rmse_wavefield,
        rmse_detector=rmse_detector,
        max_error_detector=max_error_detector,
        sample_history=recorder.sample_history,
        sample_name=sample.__class__.__name__,
        sample_params=_sample_parameters(sample),
    )
