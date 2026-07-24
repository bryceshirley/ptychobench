from ptychobench_3d.metrics import calculate_farfield_wave
import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from matplotlib.widgets import Slider

from ptychobench_3d.grid import SimulationGrid

# Configure the root logger for the entire application upon import
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger("ptychobench")


@dataclass
class BenchmarkResult:
    """
    Holds the complete history and error metrics for a single 3D simulation run.
    """

    # --- Error Metrics (RMSE relative to Exact) ---
    rmse_wavefield: dict[str, float]
    rmse_detector: dict[str, float]
    max_error_detector: dict[str, float]

    # --- Simulation Metadata ---
    grid: SimulationGrid

    # --- Propagation Data and Sample History ---
    # wavefield_history shape: (Nz, N, N)
    # sample_history shape: (Nz, N, N)
    wavefield_history: dict[str, np.ndarray] = field(repr=False)
    sample_history: np.ndarray = field(repr=False)

    def print_summary(self):
        """Prints a clean, readable summary of the benchmark run."""

        print("\nWavefield Errors (RMSE relative to Exact_mf):")
        if not self.rmse_wavefield:
            print("  No errors calculated (Exact_mf missing).")
        else:
            for name, err in sorted(
                self.rmse_wavefield.items(), key=lambda item: item[1]
            ):
                if name != "Exact_mf":
                    print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

        print("\nDetector Errors (RMSE relative to Exact_mf):")
        if not self.rmse_detector:
            print("  No errors calculated (Exact_mf missing).")
        else:
            for name, err in sorted(
                self.rmse_detector.items(), key=lambda item: item[1]
            ):
                if name != "Exact_mf":
                    print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

        print("\nDetector Errors (Max Error relative to Exact_mf):")
        if not self.max_error_detector:
            print("  No errors calculated (Exact_mf missing).")
        else:
            for name, err in sorted(
                self.max_error_detector.items(), key=lambda item: item[1]
            ):
                if name != "Exact_mf":
                    print(f"  {name:<20}: {err:.6e}")
        print("-" * 45)

    def plot_sample(self):
        """Plots the X-Z longitudinal cross-section of the sample at center Y."""
        grid = self.grid
        y_mid = grid.N // 2
        extent = [grid.x[0], grid.x[-1], grid.z_prop, 0]

        # Extract the center Y-slice: (Nz, N)
        sample_slice = self.sample_history[:, :, y_mid]

        fig_env, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        im_real = ax_real.imshow(
            np.abs(sample_slice), extent=extent, aspect="auto", cmap="viridis"
        )
        ax_real.set_title("Modulus of Sample (Center Y-Slice)", fontsize=14)
        ax_real.set_xlabel("Transverse coordinate X", fontsize=12)
        ax_real.set_ylabel("Propagation Distance Z", fontsize=12)
        fig_env.colorbar(im_real, ax=ax_real)

        im_imag = ax_imag.imshow(
            np.angle(sample_slice), extent=extent, aspect="auto", cmap="twilight"
        )
        ax_imag.set_title("Phase of Sample (radians)", fontsize=14)
        ax_imag.set_xlabel("Transverse coordinate X", fontsize=12)
        fig_env.colorbar(im_imag, ax=ax_imag)

        fig_env.suptitle("Refractive Index Environment", fontsize=16)
        return fig_env

    def plot_wavefields(self):
        """Plots the X-Z longitudinal evolution of the wavefield at center Y."""
        grid = self.grid
        y_mid = grid.N // 2
        extent = [grid.x[0], grid.x[-1], grid.z_prop, 0]

        n_ops = len(self.wavefield_history)
        fig_p, axes_p = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)
        fig_a, axes_a = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)

        if n_ops == 1:
            axes_p, axes_a = [axes_p], [axes_a]

        for i, (name, data) in enumerate(self.wavefield_history.items()):
            # Slice down to (Nz, N)
            slice_data = data[:, :, y_mid]

            im_p = axes_p[i].imshow(
                np.unwrap(np.angle(slice_data), axis=0),
                extent=extent,
                aspect="auto",
                cmap="twilight",
            )
            axes_p[i].set_title(name, fontsize=14)
            axes_p[i].set_xlabel("Transverse X", fontsize=12)

            im_a = axes_a[i].imshow(
                np.abs(slice_data), extent=extent, aspect="auto", cmap="magma"
            )
            axes_a[i].set_title(name, fontsize=14)
            axes_a[i].set_xlabel("Transverse X", fontsize=12)

        axes_p[0].set_ylabel("Propagation Distance Z", fontsize=12)
        fig_p.colorbar(im_p, ax=axes_p, fraction=0.02, pad=0.02)
        fig_p.suptitle("Longitudinal Field Evolution (Phase, Y-center)", fontsize=16)

        axes_a[0].set_ylabel("Propagation Distance Z", fontsize=12)
        fig_a.colorbar(im_a, ax=axes_a, fraction=0.02, pad=0.02)
        fig_a.suptitle(
            "Longitudinal Field Evolution (Amplitude, Y-center)", fontsize=16
        )

        return fig_p, fig_a

    def plot_exitwave_2D(
        self, comparison_wave=None, comparison_name="Comparison (abtem)"
    ):
        """Plots the full 2D (X, Y) final exit wave at Z = z_prop.

        Args:
            comparison_wave (np.ndarray, optional): An additional 2D wavefield to plot alongside the history.
            comparison_name (str, optional): The title for the comparison wave column.
        """
        grid = self.grid
        extent = [grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]]

        # Prepare the data to plot (extract final z-step from history)
        plot_data = []
        # Append the comparison wave if provided
        if comparison_wave is not None:
            plot_data.append((comparison_name, comparison_wave))
        for name, data in self.wavefield_history.items():
            plot_data.append((name, data[-1, :, :]))  # Take final z-step (N, N)

        n_plots = len(plot_data)

        # Dynamically size the figure based on total number of plots
        fig, axes = plt.subplots(
            2, n_plots, figsize=(5 * n_plots, 8), sharex=True, sharey=True
        )

        if n_plots == 1:
            axes = axes[:, np.newaxis]

        for i, (name, exit_wave) in enumerate(plot_data):
            # Amplitude (Top Row)
            im_a = axes[0, i].imshow(
                np.abs(exit_wave).T, extent=extent, origin="lower", cmap="magma"
            )
            axes[0, i].set_title(f"{name}\nAmplitude")
            if i == 0:
                axes[0, i].set_ylabel("Transverse Y")

            # Phase (Bottom Row)
            im_p = axes[1, i].imshow(
                np.angle(exit_wave).T, extent=extent, origin="lower", cmap="twilight"
            )
            axes[1, i].set_title("Phase")
            axes[1, i].set_xlabel("Transverse X")
            if i == 0:
                axes[1, i].set_ylabel("Transverse Y")

        fig.colorbar(im_a, ax=axes[0, :], pad=0.02)
        fig.colorbar(im_p, ax=axes[1, :], pad=0.02)
        fig.suptitle("2D Exit Waves at Sample Exit", fontsize=16)

        return fig

    def plot_farfield_2D(
        self,
        mode="modulus",
        log_scale: bool = True,
        error_map: bool = False,
        comparison_wave=None,
        comparison_name="Comparison (abtem)",
    ):
        """Plots the full 2D (Kx, Ky) farfield distribution.

        Args:
            mode: "modulus", "intensity", etc.
            log_scale: Whether to use LogNorm.
            error_map: If True, plots difference relative to Exact_mf.
            comparison_wave (np.ndarray, optional): An additional 2D wavefield to plot alongside the history.
            comparison_name (str, optional): The title for the comparison wave column.
        """
        from matplotlib.colors import LogNorm

        grid = self.grid
        extent = [
            grid.kx.min(),
            grid.kx.max(),
            grid.ky.min(),
            grid.ky.max(),
        ]

        # Prepare the data to plot (extract final z-step from history)
        plot_data = []
        # Append the comparison wave if provided
        if comparison_wave is not None:
            plot_data.append((comparison_name, comparison_wave))

        for name, data in self.wavefield_history.items():
            plot_data.append((name, data[-1, :, :]))

        n_plots = len(plot_data)

        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(5 * n_plots, 5),
            sharex=True,
            sharey=True,
        )

        if n_plots == 1:
            axes = [axes]
        elif n_plots > 1 and not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        if error_map:
            assert (
                "Exact_mf" in self.wavefield_history
            ), "Exact_mf data is required for error map."

            exact_exit_wave = self.wavefield_history["Exact_mf"][-1]
            exact_farfield = calculate_farfield_wave(
                exact_exit_wave,
                mode,
            )

        # --------------------------------------------------
        # Build all images first so we can determine a
        # consistent color scale.
        # --------------------------------------------------

        farfields = []

        for name, exit_wave in plot_data:
            farfield = calculate_farfield_wave(exit_wave, mode)

            if error_map and name != "Exact_mf":
                farfield = np.abs(farfield - exact_farfield)

            elif error_map and name == "Exact_mf":
                farfield = np.zeros_like(farfield)

            farfield = np.abs(np.asarray(farfield))
            farfield = np.nan_to_num(
                farfield,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            farfields.append(farfield)

        # --------------------------------------------------
        # Global normalization
        # --------------------------------------------------

        all_values = np.concatenate([f.ravel() for f in farfields])

        positive_values = all_values[all_values > 0]

        if log_scale and positive_values.size > 0:
            vmin = positive_values.min()
            vmax = positive_values.max()

            if vmax <= vmin:
                vmax = vmin * 10.0

            norm = LogNorm(vmin=vmin, vmax=vmax)

        else:
            vmin = all_values.min()
            vmax = all_values.max()

            if vmax <= vmin:
                vmax = vmin + 1e-12

            norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        im = None

        for i, ((name, _), farfield) in enumerate(zip(plot_data, farfields)):
            im = axes[i].imshow(
                farfield.T,
                extent=extent,
                origin="lower",
                cmap="viridis",
                norm=norm,
                aspect="auto",
            )

            axes[i].set_title(name)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        fig.suptitle(
            f"2D Farfield {mode.capitalize()}",
            fontsize=16,
        )

        if im is not None:
            fig.colorbar(
                im,
                ax=axes,
                fraction=0.02,
                pad=0.02,
            )

        return fig

    def plot_evolution_slider(
        self, x: float = 0.0, y: float = 0.0, abtem_field: np.ndarray = None
    ):
        """
        Interactive dual-slider for inspecting 1D wavefield evolution
        along Z at a selected (X, Y) pixel.
        """
        grid = self.grid
        styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]

        def _coord_to_index(coord: float, axis_array: np.ndarray) -> int:
            idx = int(np.argmin(np.abs(axis_array - coord)))
            return int(np.clip(idx, 0, len(axis_array) - 1))

        x_idx = _coord_to_index(x, grid.x)
        y_idx = _coord_to_index(y, grid.y)
        x_actual, y_actual = grid.x[x_idx], grid.y[y_idx]

        fig, (ax_phase, ax_amp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Leave room at the bottom for TWO sliders
        fig.subplots_adjust(bottom=0.25, hspace=0.28)

        # Combine history and abtem_field into a single dictionary for clean updating
        plot_data = self.wavefield_history.copy()
        if abtem_field is not None:
            if hasattr(abtem_field, "compute"):
                abtem_field = abtem_field.compute()
            plot_data["abtem (Reference)"] = np.asarray(abtem_field)

        phase_lines = {}
        amp_lines = {}
        style_idx = 0

        for name, data in plot_data.items():
            # Slice at specific (X, Y) to get a 1D Z array
            phase_data = np.unwrap(np.angle(data[:, x_idx, y_idx]))
            amp_data = np.abs(data[:, x_idx, y_idx])

            if name == "Exact_mf":
                (p_line,) = ax_phase.plot(
                    grid.z_steps, phase_data, "k-", lw=4, alpha=0.3, label=name
                )
                (a_line,) = ax_amp.plot(
                    grid.z_steps, amp_data, "k-", lw=4, alpha=0.3, label=name
                )
            elif name == "abtem (Reference)":
                (p_line,) = ax_phase.plot(
                    grid.z_steps, phase_data, "k--", lw=3, alpha=0.7, label=name
                )
                (a_line,) = ax_amp.plot(
                    grid.z_steps, amp_data, "k--", lw=3, alpha=0.7, label=name
                )
            else:
                style = styles[style_idx % len(styles)]
                (p_line,) = ax_phase.plot(
                    grid.z_steps, phase_data, style, lw=2, label=name
                )
                (a_line,) = ax_amp.plot(grid.z_steps, amp_data, style, lw=2, label=name)
                style_idx += 1

            phase_lines[name] = p_line
            amp_lines[name] = a_line

        ax_phase.set_title(
            f"Phase Evolution along Z at (X={x_actual:.2g}, Y={y_actual:.2g})",
            fontsize=14,
        )
        ax_phase.set_ylabel("Phase / radians", fontsize=12)
        ax_phase.grid(True, ls="--", alpha=0.7)
        ax_phase.legend(fontsize=10)

        ax_amp.set_title(
            f"Amplitude Evolution along Z at (X={x_actual:.2g}, Y={y_actual:.2g})",
            fontsize=14,
        )
        ax_amp.set_xlabel("Propagation Distance Z", fontsize=12)
        ax_amp.set_ylabel(r"Amplitude $|\psi|$", fontsize=12)
        ax_amp.grid(True, ls="--", alpha=0.7)

        # Build X and Y Sliders
        slider_ax_x = fig.add_axes([0.15, 0.12, 0.70, 0.035])
        slider_ax_y = fig.add_axes([0.15, 0.05, 0.70, 0.035])

        x_slider = Slider(
            slider_ax_x, "X", grid.x[0], grid.x[-1], valinit=x_actual, valstep=grid.dx
        )
        y_slider = Slider(
            slider_ax_y, "Y", grid.y[0], grid.y[-1], valinit=y_actual, valstep=grid.dy
        )

        def _update(val):
            xi = _coord_to_index(x_slider.val, grid.x)
            yi = _coord_to_index(y_slider.val, grid.y)
            xa, ya = grid.x[xi], grid.y[yi]

            for name, data in plot_data.items():
                phase_lines[name].set_ydata(np.unwrap(np.angle(data[:, xi, yi])))
                amp_lines[name].set_ydata(np.abs(data[:, xi, yi]))

            ax_phase.set_title(
                f"Phase Evolution along Z at (X={xa:.2g}, Y={ya:.2g})", fontsize=14
            )
            ax_amp.set_title(
                f"Amplitude Evolution along Z at (X={xa:.2g}, Y={ya:.2g})", fontsize=14
            )
            ax_phase.relim()
            ax_phase.autoscale_view()
            ax_amp.relim()
            ax_amp.autoscale_view()
            fig.canvas.draw_idle()

        x_slider.on_changed(_update)
        y_slider.on_changed(_update)

        widgets = {"x_slider": x_slider, "y_slider": y_slider}
        fig._ptychobench_widgets = widgets

        return fig, widgets
