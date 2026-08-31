"""
What a benchmark run produced.

Histories, error metrics, and the metadata needed to interpret them. This
module is the run's *record*. Drawing it lives in
:mod:`ptychobench.plotting` and saving it in :mod:`ptychobench.reporting`; the
``plot_*`` and :meth:`BenchmarkResult.generate_report` methods below are
one-line delegators kept so existing notebooks and the user guides carry on
working unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import numpy as np

from ptychobench.grid import SimulationGrid
from ptychobench.reporting import BenchmarkReport

if TYPE_CHECKING:  # pragma: no cover - only needed for the return annotations
    from matplotlib.figure import Figure
    from matplotlib.widgets import Slider

# A library configures a logger, never the root handler: how log records are
# formatted and where they go is the calling application's decision. See the
# NullHandler in ptychobench/__init__.py, and the first cell of
# benchmark_tutorial.ipynb for the caller-side basicConfig this replaced.
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Holds the complete history and error metrics for a single simulation run.

    Every per-operator dictionary here shares one key space: the operator's
    **class name**, e.g. ``"ParaxialOperator"``. Human-readable labels live in
    :attr:`display_names` and nowhere else.

    The histories are host NumPy ``complex128`` whatever backend the run
    executed on -- see :func:`ptychobench.benchmark.run_benchmark` -- so
    everything reading them can stay host code.
    """

    #: The class name of the operator the errors are measured against.
    REFERENCE: ClassVar[str] = "GroundTruthOperator"

    # --- Error Metrics (RMSE relative to the reference) ---
    # Keyed by the Operator CLASS NAME (e.g., "ParaxialOperator")
    rmse_wavefield: dict[str, float]
    rmse_detector: dict[str, float]
    max_error_detector: dict[str, float]

    # --- 2. Simulation Metadata ---
    sample_name: str
    sample_params: dict
    grid: SimulationGrid

    # --- 3. Propagation Data and Sample History ---
    wavefield_history: dict[str, np.ndarray] = field(repr=False)
    sample_history: np.ndarray = field(repr=False)

    #: Class name -> plot label, e.g. ``{"FeitFleckOperator": "Feit/Fleck"}``.
    #: Optional: a result built without it labels operators by class name.
    display_names: dict[str, str] = field(default_factory=dict)

    def label(self, class_name: str) -> str:
        """
        Return the plot label for an operator, falling back to its class name.

        Parameters
        ----------
        class_name : str
            The operator's class name.

        Returns
        -------
        str
            The display name, or ``class_name`` if none was recorded.
        """
        return self.display_names.get(class_name, class_name)

    def print_summary(self):
        """Print a readable summary of the benchmark run."""
        print(f"\n--- Benchmark Summary: {self.sample_name} ---")

        print("Parameters:")
        for param, val in self.sample_params.items():
            print(f"  {param:<12}: {val}")

        print("\n Wavefield Errors (RMSE relative to GroundTruthOperator):")
        if not self.rmse_wavefield:
            print("  No errors calculated (GroundTruthOperator missing from run).")

        # Sort errors so the smallest error is at the top
        sorted_errors = sorted(self.rmse_wavefield.items(), key=lambda item: item[1])
        for name, err in sorted_errors:
            if name != self.REFERENCE:
                print(f"  {self.label(name):<20}: {err:.6e}")
        print("-" * 45)

        print("\nDetector Intensity Errors (RMSE relative to GroundTruthOperator):")
        if not self.rmse_detector:
            print("  No errors calculated (GroundTruthOperator missing from run).")

        sorted_detector_errors = sorted(
            self.rmse_detector.items(), key=lambda item: item[1]
        )
        for name, err in sorted_detector_errors:
            if name != self.REFERENCE:
                print(f"  {self.label(name):<20}: {err:.6e}")
        print("-" * 45)

        print(
            "\nDetector Intensity Errors "
            "(Max Absolute Error relative to GroundTruthOperator):"
        )
        if not self.max_error_detector:
            print("  No errors calculated (GroundTruthOperator missing from run).")

        sorted_max_detector_errors = sorted(
            self.max_error_detector.items(), key=lambda item: item[1]
        )
        for name, err in sorted_max_detector_errors:
            if name != self.REFERENCE:
                print(f"  {self.label(name):<20}: {err:.6e}")
        print("-" * 45)

    @staticmethod
    def _lookup(errors: dict[str, float], operator_name: str, metric: str) -> float:
        """
        Look up one error by class name.

        Parameters
        ----------
        errors : dict of str to float
            The metric dictionary to read.
        operator_name : str
            The operator's class name.
        metric : str
            Named in the error message so the caller knows which dictionary
            was searched.

        Returns
        -------
        float
            The recorded error.

        Raises
        ------
        KeyError
            If ``operator_name`` was not part of the run, naming what was.

        Notes
        -----
        These accessors used to return ``float("inf")`` for a name that was not
        in the run -- a wrong number wearing the costume of a right one, since
        it formats, sorts and plots exactly like a real error. A misspelled
        operator surfaced as "infinitely bad" rather than as a typo.
        """
        try:
            return errors[operator_name]
        except KeyError:
            raise KeyError(
                f"no operator {operator_name!r} in this run's {metric}; "
                f"available: {sorted(errors)}"
            ) from None

    def get_rmse_wavefield(
        self, operator_name: Optional[str] = None
    ) -> Union[float, dict[str, float]]:
        """
        Get the wavefield RMSE, for one operator or for all of them.

        Parameters
        ----------
        operator_name : str, optional
            An operator's class name. Omit it for the whole dictionary.

        Returns
        -------
        float or dict of str to float
            The one error, or every error keyed by class name.

        Raises
        ------
        KeyError
            If ``operator_name`` was not part of the run.
        """
        if operator_name is None:
            return self.rmse_wavefield
        return self._lookup(self.rmse_wavefield, operator_name, "rmse_wavefield")

    def get_rmse_detector(
        self, operator_name: Optional[str] = None
    ) -> Union[float, dict[str, float]]:
        """
        Get the detector RMSE, for one operator or for all of them.

        Parameters
        ----------
        operator_name : str, optional
            An operator's class name. Omit it for the whole dictionary.

        Returns
        -------
        float or dict of str to float
            The one error, or every error keyed by class name.

        Raises
        ------
        KeyError
            If ``operator_name`` was not part of the run.
        """
        if operator_name is None:
            return self.rmse_detector
        return self._lookup(self.rmse_detector, operator_name, "rmse_detector")

    def get_rmse_wavefield_dict(self) -> dict[str, float]:
        """
        Get every wavefield error, for external analysis.

        Returns
        -------
        dict of str to float
            The errors, keyed by class name.
        """
        return self.rmse_wavefield

    def get_rmse_detector_dict(self) -> dict[str, float]:
        """
        Get every detector error, for external analysis.

        Returns
        -------
        dict of str to float
            The errors, keyed by class name.
        """
        return self.rmse_detector

    # -- delegators ---------------------------------------------------------
    #
    # Drawing and saving are separate concerns living in separate modules, but
    # `result.plot_wavefields()` is what the notebook and all five user guides
    # call. These keep that spelling working; each is the one-line forward to
    # the real implementation.
    #
    # `plotting` is imported per method rather than at module scope, matching
    # `Sample`'s delegators, because importing it pulls in Matplotlib. This
    # module is re-exported from the package root, so a module-scope import
    # would make `import ptychobench` -- and with it every headless script that
    # only wants an array back -- pay for the plotting stack.

    def plot_sample(self) -> "Figure":
        """
        Plot the sample's refractive index.

        Returns
        -------
        matplotlib.figure.Figure
            See :func:`ptychobench.plotting.plot_sample`.
        """
        from . import plotting

        return plotting.plot_sample(self)

    def plot_wavefields(self) -> tuple["Figure", "Figure"]:
        """
        Plot each operator's propagated field.

        Returns
        -------
        tuple of matplotlib.figure.Figure
            See :func:`ptychobench.plotting.plot_wavefields`.
        """
        from . import plotting

        return plotting.plot_wavefields(self)

    def plot_evolution_1D(self, x: float = 0.0) -> tuple["Figure", "Figure"]:
        """
        Plot phase and amplitude along z at one transverse position.

        Parameters
        ----------
        x : float, optional
            The transverse position. Defaults to ``0.0``.

        Returns
        -------
        tuple of matplotlib.figure.Figure
            See :func:`ptychobench.plotting.plot_evolution_1D`.
        """
        from . import plotting

        return plotting.plot_evolution_1D(self, x)

    def plot_evolution_slider(
        self, x: float = 0.0
    ) -> tuple["Figure", dict[str, "Slider"]]:
        """
        Plot the interactive version of :meth:`plot_evolution_1D`.

        Parameters
        ----------
        x : float, optional
            The initial transverse position. Defaults to ``0.0``.

        Returns
        -------
        tuple
            Figure and widgets; see
            :func:`ptychobench.plotting.plot_evolution_slider`.
        """
        from . import plotting

        return plotting.plot_evolution_slider(self, x)

    def plot_farfield_error(
        self,
        log_scale: bool = False,
        mode: str = "intensity",
        operators: list[str] | None = None,
    ) -> "Figure":
        """
        Plot the far-field error against the reference.

        Parameters
        ----------
        log_scale : bool, optional
            Logarithmic y-axis. Defaults to linear.
        mode : {"intensity", "magnitude", "phase"}, optional
            Which far-field quantity to difference.
        operators : list of str, optional
            Class names to include. All of them by default.

        Returns
        -------
        matplotlib.figure.Figure
            See :func:`ptychobench.plotting.plot_farfield_error`.
        """
        from . import plotting

        return plotting.plot_farfield_error(self, log_scale, mode, operators)

    def generate_report(self, save_dir: str | Path = "results") -> Path:
        """
        Write figures and text summaries to a timestamped directory.

        See :class:`ptychobench.reporting.BenchmarkReport`, which this creates.

        Parameters
        ----------
        save_dir : str or pathlib.Path, optional
            The parent directory the timestamped run directory is created in.
            Defaults to ``"results"``.

        Returns
        -------
        pathlib.Path
            The directory written to.
        """
        return BenchmarkReport(save_dir).write(self)
