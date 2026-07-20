from ptychobench.metrics import calculate_farfield_wave
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from matplotlib.widgets import Slider
from typing import Optional, Union

from ptychobench.grid import SimulationGrid

# Configure the root logger for the entire application upon import
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)

# Export this logger so scripts.py (and other files) can import and use it
logger = logging.getLogger("ptychobench")


@dataclass
class BenchmarkResult:
    """
    Holds the complete history and error metrics for a single simulation run.
    """

    # --- Error Metrics (RMSE relative to Exact) ---
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

    def print_summary(self):
        """Prints a clean, readable summary of the benchmark run for the students."""
        print(f"\n--- Benchmark Summary: {self.sample_name} ---")

        print("Parameters:")
        for param, val in self.sample_params.items():
            print(f"  {param:<12}: {val}")

        print("\n Wavefield Errors (RMSE relative to ExactOperator):")
        if not self.rmse_wavefield:
            print("  No errors calculated (ExactOperator missing from run).")

        # Sort errors so the smallest error is at the top
        sorted_errors = sorted(self.rmse_wavefield.items(), key=lambda item: item[1])
        for name, err in sorted_errors:
            if name != "ExactOperator":
                print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

        print("\nDetector Intensity Errors (RMSE relative to ExactOperator):")
        if not self.rmse_detector:
            print("  No errors calculated (ExactOperator missing from run).")

        sorted_detector_errors = sorted(
            self.rmse_detector.items(), key=lambda item: item[1]
        )
        for name, err in sorted_detector_errors:
            if name != "ExactOperator":
                print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

        print(
            "\nDetector Intensity Errors (Max Absolute Error relative to ExactOperator):"
        )
        if not self.max_error_detector:
            print("  No errors calculated (ExactOperator missing from run).")

        sorted_max_detector_errors = sorted(
            self.max_error_detector.items(), key=lambda item: item[1]
        )
        for name, err in sorted_max_detector_errors:
            if name != "ExactOperator":
                print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

    def get_rmse_wavefield(
        self, operator_name: Optional[str] = None
    ) -> Union[float, dict[str, float]]:
        """Returns the wavefield RMSE for a specific operator by its class name."""
        if operator_name is None:
            return self.rmse_wavefield
        return self.rmse_wavefield.get(operator_name, float("inf"))

    def get_rmse_detector(
        self, operator_name: Optional[str] = None
    ) -> Union[float, dict[str, float]]:
        """Returns the detector RMSE for a specific operator by its class name."""
        if operator_name is None:
            return self.rmse_detector
        return self.rmse_detector.get(operator_name, float("inf"))

    def get_rmse_wavefield_dict(self) -> dict[str, float]:
        """Returns the dictionary of wavefield errors for external analysis."""
        return self.rmse_wavefield

    def get_rmse_detector_dict(self) -> dict[str, float]:
        """Returns the dictionary of detector errors for external analysis."""
        return self.rmse_detector

    def plot_sample(self):
        """Plots the samples refractive index"""
        grid = self.grid
        extent = [-grid.L / 2, grid.L / 2, grid.z_prop, 0]

        fig_env, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        im_real = ax_real.imshow(
            np.abs(self.sample_history), extent=extent, aspect="auto", cmap="viridis"
        )
        ax_real.set_title(r"Modulus of Sample", fontsize=14)
        ax_real.set_xlabel("Transverse coordinate x", fontsize=12)
        ax_real.set_ylabel("Propagation Distance z", fontsize=12)
        fig_env.colorbar(im_real, ax=ax_real)

        im_imag = ax_imag.imshow(
            np.angle(self.sample_history), extent=extent, aspect="auto", cmap="twilight"
        )
        ax_imag.set_title(r"Phase of Sample (radians)", fontsize=14)
        ax_imag.set_xlabel("Transverse coordinate x", fontsize=12)
        fig_env.colorbar(im_imag, ax=ax_imag)

        fig_env.suptitle("Refractive Index Environment", fontsize=16)
        return fig_env

    def plot_wavefields(self):
        """Plots the propagated wavefields (phase and amplitude) for each operator."""
        grid = self.grid
        extent = [-grid.L / 2, grid.L / 2, grid.z_prop, 0]

        # 2. 2D Field Propagation (Phase and Amplitude)
        n_ops = len(self.wavefield_history)
        fig_p, axes_p = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)
        fig_a, axes_a = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)

        if n_ops == 1:
            axes_p, axes_a = [axes_p], [axes_a]

        for i, (name, data) in enumerate(self.wavefield_history.items()):
            im_p = axes_p[i].imshow(
                np.unwrap(np.angle(data), axis=0),
                extent=extent,
                aspect="auto",
                cmap="twilight",
            )
            axes_p[i].set_title(name, fontsize=14)
            axes_p[i].set_xlabel("Transverse coordinate x", fontsize=12)

            im_a = axes_a[i].imshow(
                np.abs(data), extent=extent, aspect="auto", cmap="magma"
            )
            axes_a[i].set_title(name, fontsize=14)
            axes_a[i].set_xlabel("Transverse coordinate x", fontsize=12)

        axes_p[0].set_ylabel("Propagation Distance z", fontsize=12)
        fig_p.colorbar(im_p, ax=axes_p, fraction=0.02, pad=0.02)
        fig_p.suptitle("2D Field Propagation (Phase)", fontsize=16)

        axes_a[0].set_ylabel("Propagation Distance z", fontsize=12)
        fig_a.colorbar(im_a, ax=axes_a, fraction=0.02, pad=0.02)
        fig_a.suptitle("2D Field Propagation (Amplitude)", fontsize=16)

        return fig_p, fig_a

    def plot_evolution_1D(self, x=0.0):
        """
        Plots the evolution of the wavefield along the propagation direction at a specific transverse position x.

        Parameters
        ----------
        x : float
            The transverse position (in the same units as grid.L) at which to extract the 1D cross-section.
            If x is outside the range of -grid.L/2 to grid.L/2, it defaults to the center of the grid.
        """

        grid = self.grid
        styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]

        # Determine the index corresponding to the specified x value between
        # -grid.L/2 and grid.L/2. If x is outside this range, default to the center.
        if -grid.L / 2 <= x <= grid.L / 2:
            mid_index = int((x + grid.L / 2) / grid.L * (grid.N - 1))
        else:
            raise ValueError(
                f"x={x} is out of bounds. It must be between {-grid.L / 2} and {grid.L / 2}."
            )

        # Helper for 1D line plots
        def _plot_1d_comparison(data_extractor, x_axis, xlabel, ylabel, title):
            fig = plt.figure(figsize=(10, 6))

            style_idx = 0
            for name, data in self.wavefield_history.items():
                y_data = data_extractor(data)

                if name == "Exact":
                    plt.plot(x_axis, y_data, "k-", linewidth=4, alpha=0.3, label=name)
                else:
                    style = styles[style_idx % len(styles)]
                    plt.plot(x_axis, y_data, style, linewidth=2, label=name)
                    style_idx += 1

            plt.title(title, fontsize=14)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.legend(fontsize=11, loc="best")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            return fig

        # 3. 1D Cross Sections
        fig1 = _plot_1d_comparison(
            lambda d: np.unwrap(np.angle(d[:, mid_index]), axis=0),
            grid.z_steps,
            "Propagation Distance z",
            "Phase (radians)",
            f"Phase Evolution along z at x = {x:.2f}",
        )

        fig2 = _plot_1d_comparison(
            lambda d: np.abs(d[:, mid_index]),
            grid.z_steps,
            "Propagation Distance z",
            r"Amplitude $|\psi|$",
            f"Amplitude Evolution along z at x = {x:.2f}",
        )

        return fig1, fig2

    def generate_report(self, save_dir: str = "results"):
        """
        Generates all 2D and 1D plots based on the simulation results.
        Creates a time-stamped sub-directory to prevent overwriting old runs.
        Outputs setup parameters and quantitative errors to separate text files.
        """
        # Extract the grid from the result object automatically
        grid = self.grid

        # 1. Setup the time-stamped directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = Path.cwd() / save_dir / f"run_{self.sample_name}_{timestamp}"
        save_path.mkdir(parents=True, exist_ok=True)

        # 2. Compile the Setup Information (Simulation Summary)
        summary_lines = [
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
            "\n--- Sample Setup ---",
            f"Sample:    {self.sample_name}",
        ]

        # Dynamically add all provided sample parameters
        for key, value in self.sample_params.items():
            summary_lines.append(f"{key:<12}: {value}")

        summary_file = save_path / "simulation_summary.txt"
        summary_file.write_text("\n".join(summary_lines) + "\n")

        # 3. Output quantitative errors as a table
        operator_names = sorted(
            set(self.rmse_wavefield.keys()) | set(self.rmse_detector.keys())
        )

        error_lines = [
            f"# Quantitative Errors at z = {grid.z_prop}",
            "",
            "| Operator | Wavefield RMSE | Detector Intensity RMSE |",
            "|---|---:|---:|",
        ]

        for name in operator_names:
            if name == "ExactOperator":
                continue

            wavefield_error = self.rmse_wavefield.get(name, None)
            detector_error = self.rmse_detector.get(name, None)

            wavefield_str = (
                f"{wavefield_error:.6e}" if wavefield_error is not None else "—"
            )
            detector_str = (
                f"{detector_error:.6e}" if detector_error is not None else "—"
            )

            error_lines.append(f"| {name} | {wavefield_str} | {detector_str} |")

        error_file = save_path / "errors.md"
        error_file.write_text("\n".join(error_lines) + "\n")

        # 4. Generate Visuals
        logger.info(f"Generating plots in '{save_path}'...")

        # Plot the sample's refractive index environment (modulus and phase)
        fig_env = self.plot_sample()
        fig_env.savefig(
            save_path / f"{self.sample_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig_env)

        # Plot the propagated wavefields (phase and amplitude) for each operator
        fig_p, fig_a = self.plot_wavefields()
        fig_a.savefig(
            save_path / "propagated_fields_2D_amp.png", dpi=300, bbox_inches="tight"
        )
        fig_p.savefig(
            save_path / "propagated_fields_2D_phase.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig_p)
        plt.close(fig_a)

        # Plot the evolution of the wavefield along the propagation direction at the center of the grid
        fig1, fig2 = self.plot_evolution_1D()
        fig1.savefig(save_path / "phase_evolution_1D.png", dpi=300)
        fig2.savefig(save_path / "amplitude_evolution_1D.png", dpi=300)
        plt.close(fig1)
        plt.close(fig2)

        # 5. Generate Farfield Error Plots (Intensity, Magnitude, Phase)
        fig_intensity = self.plot_farfield_error(log_scale=True, mode="intensity")
        fig_magnitude = self.plot_farfield_error(log_scale=True, mode="magnitude")
        fig_phase = self.plot_farfield_error(log_scale=True, mode="phase")
        fig_intensity.savefig(save_path / "farfield_error_intensity.png", dpi=300)
        fig_magnitude.savefig(save_path / "farfield_error_magnitude.png", dpi=300)
        fig_phase.savefig(save_path / "farfield_error_phase.png", dpi=300)
        plt.close(fig_intensity)
        plt.close(fig_magnitude)
        plt.close(fig_phase)

        logger.info("Plotting complete.")

    def plot_exitwave_error(self, log_scale: bool = False):
        """
        Plots the error of the farfield of each operator
        Assumes the exact operator has been used.

        Parameters
        ----------
        log_scale : bool
            If True, the Y-axis will be logarithmic. If False, it will be linear
        """
        assert "Exact" in self.wavefield_history.keys()
        exact_exit_wave = self.wavefield_history["Exact"][-1, :]
        grid = self.grid

        x_values = np.linspace(-grid.L / 2, grid.L / 2, grid.N)
        styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (name, data) in enumerate(self.wavefield_history.items()):
            if name == "Exact":
                continue

            error = abs(exact_exit_wave - data[-1, :])

            # Y axis is logarithmic
            if log_scale:
                ax.semilogy(x_values, error, styles[i], label=name)
            else:
                ax.plot(x_values, error, styles[i], label=name)

        ax.set_xlabel("Far Field Position")
        ax.set_ylabel("Absolute Error")
        ax.legend()
        return fig

    def plot_farfield_error(
        self,
        log_scale: bool = False,
        mode: str = "intensity",
        operators: list[str] | None = None,
    ):
        """
        Plots the error of the farfield of each operator
        Assumes the exact operator has been used.
        Parameters
        ----------
        log_scale : bool, optional
            If True, the Y-axis will be logarithmic. If False, it will be linear
        mode : str, optional
            If "intensity", the intensity error will be plotted. If "magnitude",
            the magnitude error will be plotted. If "phase", the phase error
            will be plotted. Default is "intensity".
        operators : list of str, optional
            List of operator names to include in the plot. If None, all operators
            will be included. Default is None.
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure containing the plot.
        """
        assert "Exact" in self.wavefield_history.keys()
        exact_farfield_intensity = calculate_farfield_wave(
            self.wavefield_history["Exact"][-1, :], mode
        )

        grid = self.grid

        x_values = np.linspace(-grid.L / 2, grid.L / 2, grid.N)
        styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]
        fig, ax = plt.subplots(figsize=(10, 6))

        if operators is None:
            operators = list(self.wavefield_history.keys())

        for i, name in enumerate(operators):
            data = self.wavefield_history[name]
            if name == "Exact":
                continue

            error = abs(
                exact_farfield_intensity - calculate_farfield_wave(data[-1, :], mode)
            )

            # Y axis is logarithmic
            if log_scale:
                ax.semilogy(x_values, error, styles[i], label=name)
            else:
                ax.plot(x_values, error, styles[i], label=name)

        ax.set_xlabel("Far Field Position")
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"Farfield {mode.capitalize()} vs Exact {mode.capitalize()} Error")
        ax.legend()
        return fig

    def plot_evolution_slider(self, x: float = 0.0):
        """
        Interactive matplotlib slider for inspecting 1D wavefield evolution
        along z at a selected transverse coordinate x.

        This is the interactive counterpart of plot_evolution_1D(x).

        Parameters
        ----------
        x : float
            Initial transverse position.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The interactive matplotlib figure.

        widgets : dict
            Dictionary containing the slider object. Keep this reference alive
            in scripts/notebooks to prevent widget garbage collection.
        """

        grid = self.grid
        styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]

        if not (-grid.L / 2 <= x <= grid.L / 2):
            raise ValueError(
                f"x={x} is out of bounds. It must be between "
                f"{-grid.L / 2} and {grid.L / 2}."
            )

        def _x_to_index(x_value: float) -> int:
            """
            Convert transverse position x to nearest grid index.
            """
            idx = int(np.argmin(np.abs(grid.x - x_value)))
            return int(np.clip(idx, 0, grid.N - 1))

        def _get_phase(data: np.ndarray, x_idx: int) -> np.ndarray:
            """
            Phase evolution along z at fixed x index.
            """
            return np.unwrap(np.angle(data[:, x_idx]), axis=0)

        def _get_amplitude(data: np.ndarray, x_idx: int) -> np.ndarray:
            """
            Amplitude evolution along z at fixed x index.
            """
            return np.abs(data[:, x_idx])

        x_idx = _x_to_index(x)
        x_actual = grid.x[x_idx]

        # ---------------------------------------------------------
        # Figure and axes
        # ---------------------------------------------------------
        fig, (ax_phase, ax_amp) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
        )

        fig.subplots_adjust(bottom=0.16, hspace=0.28)

        phase_lines = {}
        amp_lines = {}

        style_idx = 0

        for name, data in self.wavefield_history.items():
            phase_data = _get_phase(data, x_idx)
            amplitude_data = _get_amplitude(data, x_idx)

            if name == "Exact":
                (phase_line,) = ax_phase.plot(
                    grid.z_steps,
                    phase_data,
                    "k-",
                    linewidth=4,
                    alpha=0.3,
                    label=name,
                )

                (amp_line,) = ax_amp.plot(
                    grid.z_steps,
                    amplitude_data,
                    "k-",
                    linewidth=4,
                    alpha=0.3,
                    label=name,
                )
            else:
                style = styles[style_idx % len(styles)]

                (phase_line,) = ax_phase.plot(
                    grid.z_steps,
                    phase_data,
                    style,
                    linewidth=2,
                    label=name,
                )

                (amp_line,) = ax_amp.plot(
                    grid.z_steps,
                    amplitude_data,
                    style,
                    linewidth=2,
                    label=name,
                )

                style_idx += 1

            phase_lines[name] = phase_line
            amp_lines[name] = amp_line

        # ---------------------------------------------------------
        # Axis formatting
        # ---------------------------------------------------------
        ax_phase.set_title(
            f"Phase Evolution along z at x = {x_actual:.4g}",
            fontsize=14,
        )
        ax_phase.set_ylabel("Phase / radians", fontsize=12)
        ax_phase.grid(True, linestyle="--", alpha=0.7)
        ax_phase.legend(fontsize=10, loc="best")

        ax_amp.set_title(
            f"Amplitude Evolution along z at x = {x_actual:.4g}",
            fontsize=14,
        )
        ax_amp.set_xlabel("Propagation Distance z", fontsize=12)
        ax_amp.set_ylabel(r"Amplitude $|\psi|$", fontsize=12)
        ax_amp.grid(True, linestyle="--", alpha=0.7)
        ax_amp.legend(fontsize=10, loc="best")

        # ---------------------------------------------------------
        # Slider
        # ---------------------------------------------------------
        slider_ax = fig.add_axes([0.15, 0.05, 0.70, 0.035])

        x_slider = Slider(
            ax=slider_ax,
            label="x",
            valmin=grid.x[0],
            valmax=grid.x[-1],
            valinit=x_actual,
            valstep=grid.dx,
        )

        def _update(x_value):
            x_idx_new = _x_to_index(x_value)
            x_actual_new = grid.x[x_idx_new]

            for name, data in self.wavefield_history.items():
                phase_lines[name].set_ydata(_get_phase(data, x_idx_new))
                amp_lines[name].set_ydata(_get_amplitude(data, x_idx_new))

            ax_phase.set_title(
                f"Phase Evolution along z at x = {x_actual_new:.4g}",
                fontsize=14,
            )

            ax_amp.set_title(
                f"Amplitude Evolution along z at x = {x_actual_new:.4g}",
                fontsize=14,
            )

            ax_phase.relim()
            ax_phase.autoscale_view()

            ax_amp.relim()
            ax_amp.autoscale_view()

            fig.canvas.draw_idle()

        x_slider.on_changed(_update)

        widgets = {
            "x_slider": x_slider,
        }

        # Keep widgets alive in some notebook/script environments.
        fig._ptychobench_widgets = widgets

        return fig, widgets
