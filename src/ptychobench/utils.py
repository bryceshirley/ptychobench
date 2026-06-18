# src/ptychobench/solver/utils.py
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from ptychobench.grid import SimulationGrid
from ptychobench.solver import BenchmarkResult

# Configure the root logger for the entire application upon import
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)

# Export this logger so scripts.py (and other files) can import and use it
logger = logging.getLogger("ptychobench")


def generate_results(
    grid: SimulationGrid, result: BenchmarkResult, save_dir: str = "results"
):
    """
    Generates all 2D and 1D plots based on the simulation results.
    Creates a time-stamped sub-directory to prevent overwriting old runs.
    Outputs setup parameters and quantitative errors to separate text files.
    """
    # 1. Setup the time-stamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path.cwd() / save_dir / f"run_{timestamp}"
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
        f"Sample:    {result.sample_name}",
    ]

    # Dynamically add all provided sample parameters
    for key, value in result.sample_params.items():
        summary_lines.append(f"{key:<12}: {value}")

    summary_file = save_path / "simulation_summary.txt"
    summary_file.write_text("\n".join(summary_lines) + "\n")

    # 3. Output the Quantitative Errors (Console and separate File)
    error_lines = [f"--- Propagated Field Errors at z = {grid.z_prop} m (Relative) ---"]
    for name, error in result.errors.items():
        if name != "Exact":
            error_lines.append(f"{name:<18}: {error:.6e}")

    error_file = save_path / "quantitative_errors.txt"
    error_file.write_text("\n".join(error_lines) + "\n")

    # Log the errors cleanly to the console
    logger.info(f"Propagated Field Errors at z = {grid.z_prop} m (Relative):")
    for name, error in result.errors.items():
        if name != "Exact":
            logger.info(f"  {name:<18}: {error:.6e}")

    # 4. Generate Visuals
    logger.info(f"Generating plots in '{save_path}'...")

    extent = [-grid.L / 2, grid.L / 2, grid.z_prop, 0]
    mid_index = grid.N // 2
    styles = ["m-", "r--", "g-", "b-.", "c:", "y--", "C0-"]

    # ==========================================
    # 1. Refractive Index Environment (Epsilon)
    # ==========================================
    fig_env, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    im_real = ax_real.imshow(
        np.real(result.history_eps), extent=extent, aspect="auto", cmap="viridis"
    )
    ax_real.set_title(r"Real Part of $\epsilon(x,z)$", fontsize=14)
    ax_real.set_xlabel("Transverse coordinate x (m)", fontsize=12)
    ax_real.set_ylabel("Propagation Distance z (m)", fontsize=12)
    fig_env.colorbar(im_real, ax=ax_real)  # <-- FIX: Add ax=

    im_imag = ax_imag.imshow(
        np.imag(result.history_eps), extent=extent, aspect="auto", cmap="plasma"
    )
    ax_imag.set_title(r"Imaginary Part of $\epsilon(x,z)$", fontsize=14)
    ax_imag.set_xlabel("Transverse coordinate x (m)", fontsize=12)
    fig_env.colorbar(im_imag, ax=ax_imag)  # <-- FIX: Add ax=

    fig_env.suptitle("Refractive Index Environment", fontsize=16)
    fig_env.savefig(
        save_path / "epsilon_environment_2D.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_env)

    # 2. 2D Field Propagation (Phase and Amplitude)
    n_ops = len(result.histories)
    fig_p, axes_p = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)
    fig_a, axes_a = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)

    if n_ops == 1:
        axes_p, axes_a = [axes_p], [axes_a]

    for i, (name, data) in enumerate(result.histories.items()):
        im_p = axes_p[i].imshow(
            np.angle(data), extent=extent, aspect="auto", cmap="magma"
        )
        axes_p[i].set_title(name, fontsize=14)
        axes_p[i].set_xlabel("Transverse coordinate x (m)", fontsize=12)

        im_a = axes_a[i].imshow(
            np.abs(data), extent=extent, aspect="auto", cmap="magma"
        )
        axes_a[i].set_title(name, fontsize=14)
        axes_a[i].set_xlabel("Transverse coordinate x (m)", fontsize=12)

    axes_p[0].set_ylabel("Propagation Distance z (m)", fontsize=12)
    fig_p.colorbar(im_p, ax=axes_p, fraction=0.02, pad=0.02)
    fig_p.suptitle("2D Field Propagation (Phase)", fontsize=16)
    fig_p.savefig(
        save_path / "propagated_fields_2D_phase.png", dpi=300, bbox_inches="tight"
    )

    axes_a[0].set_ylabel("Propagation Distance z (m)", fontsize=12)
    fig_a.colorbar(im_a, ax=axes_a, fraction=0.02, pad=0.02)
    fig_a.suptitle("2D Field Propagation (Amplitude)", fontsize=16)
    fig_a.savefig(
        save_path / "propagated_fields_2D_amp.png", dpi=300, bbox_inches="tight"
    )

    plt.close(fig_p)
    plt.close(fig_a)

    # Helper for 1D line plots
    def _plot_1d_comparison(data_extractor, x_axis, xlabel, ylabel, title, filename):
        fig = plt.figure(figsize=(10, 6))

        style_idx = 0
        for name, data in result.histories.items():
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
        fig.savefig(save_path / filename, dpi=300)
        plt.close(fig)

    # 3. 1D Cross Sections
    _plot_1d_comparison(
        lambda d: np.abs(d[-1, :]),
        grid.x,
        "Transverse coordinate x (m)",
        r"Amplitude $|\psi|$",
        f"Exit Wave Function Amplitudes at z = {grid.z_prop} m",
        "exit_wave_functions_1D.png",
    )

    _plot_1d_comparison(
        lambda d: np.angle(d[-1, :]),
        grid.x,
        "Transverse coordinate x (m)",
        "Phase (radians)",
        f"Exit Wave Function Phases at z = {grid.z_prop} m",
        "exit_wave_phases_1D.png",
    )

    _plot_1d_comparison(
        lambda d: np.angle(d[:, mid_index]),
        grid.z_steps,
        "Propagation Distance z (m)",
        "Phase (radians)",
        f"Phase Evolution at x = {grid.x[mid_index]:.2f} m",
        "phase_evolution_1D.png",
    )

    _plot_1d_comparison(
        lambda d: np.abs(d[:, mid_index]),
        grid.z_steps,
        "Propagation Distance z (m)",
        r"Amplitude $|\psi|$",
        f"Amplitude Evolution at x = {grid.x[mid_index]:.2f} m",
        "amplitude_evolution_1D.png",
    )

    logger.info("Plotting complete.")
