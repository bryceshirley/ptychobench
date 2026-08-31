"""
Writing a benchmark run to disk.

Separated from :class:`~ptychobench.results.BenchmarkResult` because saving is
a different job from holding: it touches the filesystem, decides on a directory
layout, and serialises two text formats. None of that is something a result
*is*.

The output contract -- a timestamped directory containing a fixed set of
filenames -- is what the Results Guide documents and what
``tests/test_results.py`` pins, so treat the names in :meth:`BenchmarkReport.write`
as public.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from .results import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkReport:
    """
    Write a run's figures and text summaries into a timestamped directory.

    Parameters
    ----------
    save_dir : str or Path, optional
        The parent directory, relative to the current working directory.
        Defaults to ``"results"``. Each :meth:`write` creates a fresh
        ``run_<sample>_<timestamp>`` subdirectory inside it, so repeated runs
        never overwrite each other.
    """

    def __init__(self, save_dir: str | Path = "results") -> None:
        self.save_dir = Path(save_dir)

    def write(self, result: BenchmarkResult) -> Path:
        """
        Write the whole report and return the directory it landed in.

        Parameters
        ----------
        result : BenchmarkResult
            The run to serialise.

        Returns
        -------
        Path
            The freshly created ``run_<sample>_<timestamp>`` directory.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = Path.cwd() / self.save_dir / f"run_{result.sample_name}_{timestamp}"
        save_path.mkdir(parents=True, exist_ok=True)

        (save_path / "simulation_summary.txt").write_text(
            self._summary_text(result, timestamp)
        )
        (save_path / "errors.md").write_text(self._error_table(result))

        logger.info(f"Generating plots in '{save_path}'...")
        self._write_figures(result, save_path)
        logger.info("Plotting complete.")

        return save_path

    # -- serialisers --------------------------------------------------------

    @staticmethod
    def _summary_text(result: BenchmarkResult, timestamp: str) -> str:
        """
        Render the human-readable record of what was run.

        Parameters
        ----------
        result : BenchmarkResult
            The run to describe.
        timestamp : str
            The run timestamp, as it appears in the directory name.

        Returns
        -------
        str
            The summary log, newline-terminated.
        """
        grid = result.grid
        lines = [
            "========================================",
            "        SIMULATION SUMMARY LOG          ",
            "========================================",
            f"Run Date/Time: {timestamp}",
            "\n--- Grid & Physical Setup ---",
            f"divergence_angle: {grid.divergence_angle:.2f} deg",
            f"lam:         {grid.lam}",
            f"L:           {grid.L}",
            f"z_prop:      {grid.z_prop}",
            f"N (pixels):  {grid.N}",
            f"Nz (steps):  {grid.Nz}",
            f"probe_width: {grid.probe_width}",
            f"dx:          {grid.dx:.6e}",
            f"dz:          {grid.dz:.6e}",
            f"backend:     {grid.backend} on {grid.resolved_device}",
            "\n--- Sample Setup ---",
            f"Sample:    {result.sample_name}",
        ]
        lines += [f"{key:<12}: {value}" for key, value in result.sample_params.items()]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _error_table(result: BenchmarkResult) -> str:
        """
        Render the errors as a Markdown table, so the docs show it as-is.

        Parameters
        ----------
        result : BenchmarkResult
            The run whose error dictionaries are tabulated.

        Returns
        -------
        str
            The Markdown table, newline-terminated. The reference operator has
            no row, since it would be a row of zeros.
        """
        names = sorted(
            set(result.rmse_wavefield)
            | set(result.rmse_detector)
            | set(result.max_error_detector)
        )

        lines = [
            f"# Quantitative Errors at z = {result.grid.z_prop}",
            "",
            "| Operator | Wavefield RMSE | Detector Intensity RMSE "
            "| Detector Max Abs Error |",
            "|---|---:|---:|---:|",
        ]

        for name in names:
            # The reference is what the errors are measured against, so it has
            # no row of its own -- it would be a row of zeros.
            if name == result.REFERENCE:
                continue

            def fmt(value: float | None) -> str:
                return f"{value:.6e}" if value is not None else "—"

            lines.append(
                f"| {name} | {fmt(result.rmse_wavefield.get(name))} "
                f"| {fmt(result.rmse_detector.get(name))} "
                f"| {fmt(result.max_error_detector.get(name))} |"
            )

        return "\n".join(lines) + "\n"

    # -- figures ------------------------------------------------------------

    @staticmethod
    def _write_figures(result: BenchmarkResult, save_path: Path) -> None:
        """
        Draw every figure, save it, and close it.

        Parameters
        ----------
        result : BenchmarkResult
            The run to plot.
        save_path : Path
            The directory the figures are written into.

        Notes
        -----
        Closing matters: ``generate_report`` in a notebook loop would otherwise
        leave nine figures per call open, and Matplotlib warns at twenty.

        Matplotlib is imported here rather than at module scope so that
        ``import ptychobench``, which re-exports this class, does not drag in
        the plotting stack.
        """
        import matplotlib.pyplot as plt

        from . import plotting

        fig_p, fig_a = plotting.plot_wavefields(result)
        fig_phase_1d, fig_amp_1d = plotting.plot_evolution_1D(result)

        figures = {
            f"{result.sample_name}.png": plotting.plot_sample(result),
            "propagated_fields_2D_phase.png": fig_p,
            "propagated_fields_2D_amp.png": fig_a,
            "phase_evolution_1D.png": fig_phase_1d,
            "amplitude_evolution_1D.png": fig_amp_1d,
        }
        for mode in ("intensity", "magnitude", "phase"):
            figures[f"farfield_error_{mode}.png"] = plotting.plot_farfield_error(
                result, log_scale=True, mode=mode
            )

        for filename, fig in figures.items():
            fig.savefig(save_path / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
