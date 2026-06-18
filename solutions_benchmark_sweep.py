import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ptychobench.grid import SimulationGrid
from ptychobench.samples import straight, zigs, straight_balls, zig_balls
from ptychobench.operators import (
    ExactOperator,
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
)
from ptychobench.solver import run_benchmark


def run_sweeps():
    print("=== Ptychobench Solution: Parameter Sweeps ===")

    # Setup the output directory
    save_dir = Path.cwd() / "results" / "benchmark_sweeps"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define the physics to test
    samples = [straight, zigs, straight_balls, zig_balls]
    operators = [ExactOperator, ParaxialOperator, FeitFleckOperator, LinDudaOperator]

    # -----------------------------------------------------------------
    # Sweep 1: Error vs. Divergence Angle (Fixed Modulus)
    # -----------------------------------------------------------------
    print("\n--- Starting Sweep 1: Divergence Angle ---")
    fixed_modulus = 0.1
    angles = np.linspace(0.0, 70.0, 10)  # Sweep from 0 to 70 degrees

    # Create a figure with 1 row and 4 columns (one for each sample)
    fig1, axes1 = plt.subplots(1, len(samples), figsize=(20, 5), sharey=True)
    fig1.suptitle(
        f"Error vs Divergence Angle (Fixed Modulus = {fixed_modulus})", fontsize=16
    )

    for idx, sample_func in enumerate(samples):
        print(f"  Running angle sweep for {sample_func.__name__}...")

        # Dictionaries to hold the error lists for this specific sample
        q1_errs, q2_errs, q3_errs = [], [], []

        for angle in angles:
            grid = SimulationGrid(divergence_angle=angle)
            res = run_benchmark(
                grid=grid,
                sample=sample_func,
                operators=operators,
                sample_params={"modulus": fixed_modulus, "phase_pert": 0.06},
            )

            q1_errs.append(res.errors["Paraxial (Q1)"])
            q2_errs.append(res.errors["Feit/Fleck (Q2)"])
            q3_errs.append(res.errors["Lin/Duda (Q3)"])

        # Plot this sample's results on its respective subplot
        ax = axes1[idx]
        ax.plot(angles, q1_errs, "r-o", label="Paraxial (Q1)")
        ax.plot(angles, q2_errs, "g-s", label="Feit/Fleck (Q2)")
        ax.plot(angles, q3_errs, "b-^", label="Lin/Duda (Q3)")

        ax.set_title(sample_func.__name__)
        ax.set_xlabel("Divergence Angle (Degrees)")
        ax.set_yscale("log")  # Crucial for visualizing the gap between orders
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Relative Error")
            ax.legend()

    plt.tight_layout()
    fig1.savefig(save_dir / "sweep_divergence_angle.png", dpi=300)
    plt.close(fig1)

    # -----------------------------------------------------------------
    # Sweep 2: Error vs. Modulus (Fixed Divergence Angle)
    # -----------------------------------------------------------------
    print("\n--- Starting Sweep 2: Modulus ---")
    fixed_angle = 10.0
    moduli = np.linspace(0.01, 2.0, 10)  # Sweep strength of the sample

    fig2, axes2 = plt.subplots(1, len(samples), figsize=(20, 5), sharey=True)
    fig2.suptitle(
        f"Error vs Sample Modulus (Fixed Angle = {fixed_angle}°)", fontsize=16
    )

    for idx, sample_func in enumerate(samples):
        print(f"  Running modulus sweep for {sample_func.__name__}...")

        q1_errs, q2_errs, q3_errs = [], [], []

        for mod in moduli:
            grid = SimulationGrid(divergence_angle=fixed_angle)
            res = run_benchmark(
                grid=grid,
                sample=sample_func,
                operators=operators,
                sample_params={"modulus": mod, "phase_pert": 0.06},
            )

            q1_errs.append(res.errors["Paraxial (Q1)"])
            q2_errs.append(res.errors["Feit/Fleck (Q2)"])
            q3_errs.append(res.errors["Lin/Duda (Q3)"])

        ax = axes2[idx]
        ax.plot(moduli, q1_errs, "r-o", label="Paraxial (Q1)")
        ax.plot(moduli, q2_errs, "g-s", label="Feit/Fleck (Q2)")
        ax.plot(moduli, q3_errs, "b-^", label="Lin/Duda (Q3)")

        ax.set_title(sample_func.__name__)
        ax.set_xlabel("Sample Modulus (Strength)")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Relative Error")
            ax.legend()

    plt.tight_layout()
    fig2.savefig(save_dir / "sweep_modulus.png", dpi=300)
    plt.close(fig2)

    print(f"\nAll sweeps complete! Plots saved to: {save_dir}")


if __name__ == "__main__":
    run_sweeps()
