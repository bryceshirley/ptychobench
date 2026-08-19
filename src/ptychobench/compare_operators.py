"""
Issue #56: compare the trained FNO against the classical operators.

    1. Reproduce the dataset_check plots, with the FNO prediction alongside
       the exact solution and the Feit/Fleck baseline.
    2. Plot the error between the ground truth and each operator.

Run generate_data.py and then train_fno.py first: this script plots the model
that train_fno.py trained and saved, it does not train one of its own.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

from dataset import PtychoDataset
from ptychobench.grid import SimulationGrid
from train_fno import BATCH_SIZE, SPLIT_SEED, WEIGHTS_PATH, build_model

DATA_PATH = "simulation_data.pt"
SAVE_DIR = Path("scripts/results")

CONFIG = 0  # Which of the num_configs configurations to plot, overridable on the command line
Z_STEP = 0  # Which z-step of that configuration to plot


def find_rows(data, config, z_step):
    """
    Returns the row index of every sample type at one configuration and z-step.

    Reads the labels generate_data.py stored, so it does not depend on the order
    the generation loops happened to run in.
    """
    return [
        row
        for row in range(len(data["sample_names"]))
        if data["config_ids"][row] == config and data["z_indices"][row] == z_step
    ]


def load_trained_model():
    """Loads the model train_fno.py trained, so both scripts report the same model."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No weights at {WEIGHTS_PATH}. Run train_fno.py first to train and save a model."
        )

    model = build_model()
    # The FNO's state_dict stores neuralop objects, too many to allowlist one by one,
    # so fall back to full pickle. Safe here: train_fno.py wrote this file on this machine
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=False))
    print(f"Loaded the model trained by train_fno.py from {WEIGHTS_PATH}")
    return model


def predict(model, dataset, row):
    """Runs one dataset row through the FNO and rebuilds a complex wavefield."""
    features, _ = dataset[row]
    model.eval()
    with torch.no_grad():
        prediction = model(features.unsqueeze(0))[0]
    return (prediction[0] + 1j * prediction[1]).numpy()


def compare(path=DATA_PATH, config=CONFIG, z_step=Z_STEP):
    data = torch.load(path)#fetch complex wavefields and eps tensorsfor plotting
    dataset = PtychoDataset(path)#fetch real tensors for FNO input and output

    grid = SimulationGrid(**data["grid_params"])
    eps, psi_exact, psi_baseline = data["input_eps"], data["target_psi"], data["baseline_psi"]

    # Reproduce the split train_fno.py uses, obtain the 8 rows of trained dataset so each row can be labelled
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, _ = random_split(
        dataset, [BATCH_SIZE, len(dataset) - BATCH_SIZE], generator=generator
    )
    train_indices = set(train_set.indices)

    model = load_trained_model()

    rows = find_rows(data, config, z_step)
    fig, axes = plt.subplots(len(rows), 4, figsize=(25, 3.4 * len(rows)))
    fig.suptitle(f"Operator comparison - configuration {config}, z-step {z_step}", fontsize=14)

    print(f"\n{'sample':24}  {'seen?':7}  {'Feit/Fleck':>11}  {'FNO':>11}")
    for sample_type, row in enumerate(rows):
        name = data["sample_names"][row]

        exact = psi_exact[row].numpy()
        feit_fleck = psi_baseline[row].numpy()
        fno = predict(model, dataset, row)
        seen = "trained" if row in train_indices else "unseen"

        ff_error = np.abs(exact - feit_fleck)
        fno_error = np.abs(exact - fno)
        print(
            f"{name:24}  {seen:7}  "
            f"{np.linalg.norm(exact - feit_fleck) / np.linalg.norm(exact):11.4f}  "
            f"{np.linalg.norm(exact - fno) / np.linalg.norm(exact):11.4f}"
        )

        # Column 0: the sample the beam passes through
        ax = axes[sample_type, 0]
        ax.plot(grid.x, np.abs(eps[row].numpy()))
        ax.set_title(f"{name}: |eps(x)|")

        # Column 1: magnitude of each operator's prediction
        ax = axes[sample_type, 1]
        ax.plot(grid.x, np.abs(exact), label="exact")
        ax.plot(grid.x, np.abs(feit_fleck), "--", label="Feit/Fleck")
        ax.plot(grid.x, np.abs(fno), ":", label="FNO")
        ax.set_title(f"{name}: |psi|  ({seen})")
        ax.legend(fontsize=8)

        # Column 2: the phase, where most of the Feit/Fleck error lives
        ax = axes[sample_type, 2]
        ax.plot(grid.x, np.angle(exact), label="exact")
        ax.plot(grid.x, np.angle(feit_fleck), "--", label="Feit/Fleck")
        ax.plot(grid.x, np.angle(fno), ":", label="FNO")
        ax.set_title(f"{name}: arg(psi)  [rad]")
        ax.legend(fontsize=8)

        # Column 3: how far each operator is from the ground truth
        ax = axes[sample_type, 3]
        ax.semilogy(grid.x, ff_error, "--", label="|exact - Feit/Fleck|")
        ax.semilogy(grid.x, fno_error, ":", label="|exact - FNO|")
        ax.set_title(f"{name}: error against ground truth")
        ax.legend(fontsize=8)

        for col in range(4):
            axes[sample_type, col].set_xlabel("x (nm)")
            axes[sample_type, col].grid(alpha=0.3)

    fig.tight_layout()
    save_path = SAVE_DIR / f"operator_comparison_config{config}_zstep{z_step}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison to {save_path}")


if __name__ == "__main__":
    # Optionally pick the configuration and z-step: compare_operators.py CONFIG Z_STEP
    if len(sys.argv) == 3:
        compare(config=int(sys.argv[1]), z_step=int(sys.argv[2]))
    else:
        compare()
