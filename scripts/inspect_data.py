"""
Quick visual check of the dataset written by generate_data.py.

The figure is saved to scripts/results/. Pass --show to open it in a window as well.

Usage:
    python scripts/inspect_data.py [path/to/simulation_data.pt] [--show]
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from ptychobench.grid import SimulationGrid

SAVE_DIR = Path("scripts/results")

# Must match the sample list in generate_data.py, in the same order
SAMPLE_NAMES = ["Apoferritin", "StraightWaveguides", "SharpStraightWaveguides"]

CONFIG = 0  # Which of the total_samples configurations to plot
Z_STEP = 0  # Which z-step of that configuration to plot

# Name the figure after what it shows, so different settings don't overwrite each other
SAVE_PATH = SAVE_DIR / f"dataset_check_config{CONFIG}_zstep{Z_STEP}.png"


def inspect(path="simulation_data.pt", show=False):
    data = torch.load(path, weights_only=False)

    grid = SimulationGrid(**data["grid_params"])
    eps, psi_in = data["input_eps"], data["input_psi"]
    psi_exact, psi_baseline = data["target_psi"], data["baseline_psi"]

    n_types, Nz = len(SAMPLE_NAMES), grid.Nz

    # 1. Print a summary of every row in the dataset
    print(f"{len(eps)} rows of {grid.N} pixels  (Nz={Nz}, {n_types} sample types)")
    print(f"{'row':>4}  {'sample':24}  {'z-step':>6}  {'max|eps|':>9}  {'residual':>9}")
    for row in range(len(eps)):
        config, rem = divmod(row, n_types * Nz)
        sample_type, z_step = divmod(rem, Nz)
        # Relative error of the classical baseline against the exact solution
        residual = (psi_exact[row] - psi_baseline[row]).abs().norm() / psi_exact[row].abs().norm()
        print(
            f"{row:>4}  {SAMPLE_NAMES[sample_type]:24}  {z_step:>6}  "
            f"{eps[row].abs().max():>9.5f}  {residual:>9.3e}"
        )

    # 2. Plot one row per sample type: the sample itself, and the wavefields
    fig, axes = plt.subplots(n_types, 3, figsize=(19, 3.2 * n_types))
    fig.suptitle(f"Dataset check - configuration {CONFIG}, z-step {Z_STEP}")

    for sample_type, name in enumerate(SAMPLE_NAMES):
        row = CONFIG * n_types * Nz + sample_type * Nz + Z_STEP

        # Left: the refractive index perturbation the beam passes through
        ax = axes[sample_type, 0]
        ax.plot(grid.x, np.abs(eps[row].numpy()))
        ax.set_title(f"{name}: |eps(x)|")
        ax.set_xlabel("x (nm)")
        ax.grid(alpha=0.3)

        # Middle: the magnitude of the input beam, the ground truth and the baseline
        ax = axes[sample_type, 1]
        ax.plot(grid.x, np.abs(psi_in[row].numpy()), label="input psi(z)")
        ax.plot(grid.x, np.abs(psi_exact[row].numpy()), "--", label="exact psi(z+dz)")
        ax.plot(grid.x, np.abs(psi_baseline[row].numpy()), ":", label="Feit/Fleck psi(z+dz)")
        ax.set_title(f"{name}: |psi|")
        ax.set_xlabel("x (nm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right: the phase, which is where the Feit/Fleck error actually lives.
        # The magnitudes agree to a fraction of a percent, so the middle panel
        # cannot show the difference the network has to learn.
        ax = axes[sample_type, 2]
        ax.plot(grid.x, np.angle(psi_in[row].numpy()), label="input psi(z)")
        ax.plot(grid.x, np.angle(psi_exact[row].numpy()), "--", label="exact psi(z+dz)")
        ax.plot(grid.x, np.angle(psi_baseline[row].numpy()), ":", label="Feit/Fleck psi(z+dz)")
        ax.set_title(f"{name}: arg(psi)  [rad, wrapped to (-pi, pi]]")
        ax.set_xlabel("x (nm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # 3. Save the figure, and only open a window if it was asked for
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"\nSaved figure to {SAVE_PATH}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    inspect(
        args[0] if args else "simulation_data.pt",
        show="--show" in sys.argv,
    )
