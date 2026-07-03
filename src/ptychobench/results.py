from ptychobench.metrics import calculate_farfield_intensity
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

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
    errors: dict[str, float]

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

        print("\nErrors (RMSE relative to ExactOperator):")
        if not self.errors:
            print("  No errors calculated (ExactOperator missing from run).")

        # Sort errors so the smallest error is at the top
        sorted_errors = sorted(self.errors.items(), key=lambda item: item[1])
        for name, err in sorted_errors:
            if name != "ExactOperator":
                print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

    def get_error(self, operator_name: str) -> float:
        """Returns the error for a specific operator by its class name."""
        return self.errors.get(operator_name, float("inf"))

    def get_errors(self) -> dict[str, float]:
        """Returns the dictionary of errors for external analysis."""
        return self.errors

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
            np.unwrap(np.angle(self.sample_history)),
            extent=extent,
            aspect="auto",
            cmap="plasma",
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
                np.angle(data), extent=extent, aspect="auto", cmap="magma"
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

    def plot_evolution_1D(self, plot_line=0.5):  # middle index for x
        """Plots the evolution of the wavefield along the propagation direction at the center of the grid."""
        grid = self.grid
        styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]
        mid_index = grid.N * plot_line  # index for x
        mid_index = int(mid_index)
        if plot_line > 1.0 or plot_line < 0.0:
            assert False

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
            lambda d: np.angle(d[:, mid_index]),
            grid.z_steps,
            "Propagation Distance z",
            "Phase (radians)",
            f"Phase Evolution along z at x = {grid.x[mid_index]:.2f}",
        )

        fig2 = _plot_1d_comparison(
            lambda d: np.abs(d[:, mid_index]),
            grid.z_steps,
            "Propagation Distance z",
            r"Amplitude $|\psi|$",
            f"Amplitude Evolution along z at x = {grid.x[mid_index]:.2f}",
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
            f"divergence_angle: {grid.divergence_angle}",
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

        # 3. Output the Quantitative Errors (Console and separate File)
        error_lines = [f"--- Propagated Field Errors at z = {grid.z_prop} (RMSE) ---"]
        for name, error in self.errors.items():
            if name != "ExactOperator":
                error_lines.append(f"{name:<18}: {error:.6e}")

        error_file = save_path / "quantitative_errors.txt"
        error_file.write_text("\n".join(error_lines) + "\n")

        # Log the errors cleanly to the console
        logger.info(f"Propagated Field Errors at z = {grid.z_prop} (RMSE):")
        for name, error in self.errors.items():
            if name != "ExactOperator":
                logger.info(f"  {name:<18}: {error:.6e}")

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

        logger.info("Plotting complete.")

    def plot_farfield_error(self):
        """
        Plots the error of the farfield of each operator
        Assumes the exact operator has been used.
        """
        assert "Exact" in self.wavefield_history.keys()
        exact_wavefield_history = self.wavefield_history["Exact"]
        exact_farfield = exact_wavefield_history[-1, :]
        grid = self.grid

        x_values = [i for i in range(grid.N)]
        colours = ["green", "yellow", "red"]

        for name, data in self.wavefield_history.items():
            if name == "Exact":
                continue
            farfield_error = []
            for x in x_values:
                farfield_error.append(abs(exact_farfield[x] - data[-1, :][x]))

            if len(colours) > 0:
                colour = colours.pop()
            else:
                colour = "blue"

            # Y axis is logarithmic
            plt.semilogy(x_values, farfield_error, "-o", c=colour, label=name)

        plt.xlabel("Far Field Position")
        plt.ylabel("Error")
        plt.legend()

    def plot_farfield_intensity_error(self):
        """
        Plots the intensity error of the farfield of each operator
        Assumes the exact operator has been used.
        """
        assert "Exact" in self.wavefield_history.keys()
        exact_wavefield_history = self.wavefield_history["Exact"]
        exact_farfield = exact_wavefield_history[-1, :]
        exact_farfield_wave = calculate_farfield_intensity(exact_farfield)
        grid = self.grid

        x_values = [i for i in range(grid.N)]
        colours = ["green", "yellow", "red"]

        for name, data in self.wavefield_history.items():
            if name == "Exact":
                continue
            farfield_wave = calculate_farfield_intensity(data[-1, :])
            farfield_error = []
            for x in x_values:
                farfield_error.append(abs(exact_farfield_wave[x] - farfield_wave[x]))

            if len(colours) > 0:
                colour = colours.pop()
            else:
                colour = "blue"

            # Y axis is logarithmic
            # plt.semilogy(x_values, farfield_error, "-o", c=colour, label=name)

            plt.plot(x_values, farfield_error, "-o", c=colour, label=name)

        plt.xlabel("Far Field Position")
        plt.ylabel("Error")
        plt.legend()
