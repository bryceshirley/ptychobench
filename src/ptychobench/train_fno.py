"""
Step 3: train a basic FNO on the simulation data.

Run generate_data.py first, so that simulation_data.pt exists.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from neuralop.losses import LpLoss
from neuralop.models import FNO
from torch.utils.data import DataLoader, random_split

from dataset import PtychoDataset

LOSS_PLOT_PATH = Path("scripts/results/training_loss.png")
GENERALISATION_PLOT_PATH = Path("scripts/results/generalisation.png")
WEIGHTS_PATH = Path(
    "scripts/results/fno_weights.pt"
)  # compare_operators.py loads the model from here
SPLIT_SEED = (
    0  # Fixes which samples are held out, so "unseen" means the same thing every run
)
WANDB_PROJECT = (
    "ptychobench-fno"  # Every run of this script shows up under this project
)

# The channel counts come straight from what PtychoDataset serves:
#   input  [Real(psi), Imag(psi), eps]        -> 3 channels
#   target [Real(psi_exact), Imag(psi_exact)] -> 2 channels
IN_CHANNELS = 3
OUT_CHANNELS = 2

# One entry per spatial dimension, and the wavefields are 1D across x
N_MODES = (64,)  # Number of Fourier modes the operator keeps
HIDDEN_CHANNELS = 64  # Width of the internal feature representation

LEARNING_RATE = 1e-3  # The usual starting point for an FNO
BATCH_SIZE = 8  # The single batch the model is trained on
NUM_EPOCHS = 5000  # Many passes over that one batch, so it can memorise it


def build_model():
    """1. Creates a basic FNO for the 1D wavefield data."""
    return FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
    )


def build_loss():
    """
    2. Relative L2 loss, the standard choice for Fourier neural operators.

    Being relative, it normalises by the size of the target, so weak and strong
    perturbations count equally. reduction="mean" keeps the value independent of
    the batch size, and 0.1 reads as a 10% error.
    """
    return LpLoss(d=1, p=2, reduction="mean")


def build_optimizer(model):
    """2. Adam, as suggested in the issue."""
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_single_batch(
    model, loss_fn, optimizer, features, targets, num_epochs=NUM_EPOCHS
):
    """
    3. Trains on one fixed batch, giving the model every chance to memorise it.

    Returns the history of losses seen during training, one per epoch. Each entry is
    measured just before that epoch's update, so it describes the model as it was
    partway through training, not the model this function hands back. Use it to draw
    the training curve, and evaluate() to measure the finished model.
    """
    model.train()
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Gradients accumulate by default, so clear them first
        prediction = model(features)
        loss = loss_fn(
            prediction, targets
        )  # The model as it stands at the start of this epoch
        loss.backward()  # Work out how each parameter affected the loss
        optimizer.step()  # Nudge the parameters in the direction that lowers it

        losses.append(loss.item())

        # Only log if the caller started a run, so the function still works on its own
        if wandb.run is not None:
            wandb.log({"loss": loss.item()}, step=epoch)

        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"  epoch {epoch + 1:>5}/{num_epochs}   loss {loss.item():.6f}")

    return losses


def plot_losses(losses, path=LOSS_PLOT_PATH):
    """4. Plots the loss against the epoch, on a log scale as it spans two orders of magnitude."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 loss")
    ax.set_title(f"Training loss on a single batch of {BATCH_SIZE} samples")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve to {path}")


def evaluate(model, loss_fn, features, targets):
    """
    5. Returns the loss on a batch without updating the model.

    Unlike the entries in the training history, this measures the model in its final
    state, so the training and unseen scores below are taken from the same model and
    are comparable.
    """
    model.eval()
    with torch.no_grad():  # No gradients needed, this is not training
        return loss_fn(model(features), targets).item()


def plot_generalisation(
    model, train_batch, unseen_batch, path=GENERALISATION_PLOT_PATH
):
    """5. Compares predictions against the truth, for a trained sample and an unseen one."""
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    for row, (name, (features, targets)) in enumerate(
        [("Trained on", train_batch), ("Unseen", unseen_batch)]
    ):
        with torch.no_grad():
            prediction = model(features)

        for col, channel in enumerate(["Real(psi)", "Imag(psi)"]):
            ax = axes[row, col]
            ax.plot(targets[0, col].numpy(), label="exact")
            ax.plot(prediction[0, col].numpy(), "--", label="FNO prediction")
            ax.set_title(f"{name}: {channel}")
            ax.set_xlabel("x (pixel)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved generalisation plot to {path}")


if __name__ == "__main__":
    # Record the hyperparameters alongside the results, so runs can be compared later
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "n_modes": N_MODES[0],
            "hidden_channels": HIDDEN_CHANNELS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
        },
    )

    model = build_model()
    loss_fn = build_loss()
    optimizer = build_optimizer(model)

    # Hold some samples back, so there is something the model has genuinely never seen
    dataset = PtychoDataset()
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, unseen_set = random_split(
        dataset, [BATCH_SIZE, len(dataset) - BATCH_SIZE], generator=generator
    )

    # One fixed batch to train on, and one batch of everything held back
    features, targets = next(iter(DataLoader(train_set, batch_size=BATCH_SIZE)))
    unseen_features, unseen_targets = next(
        iter(DataLoader(unseen_set, batch_size=len(unseen_set)))
    )

    print(
        f"Training on a single batch of {len(features)} samples for {NUM_EPOCHS} epochs"
    )
    print(f"Holding back {len(unseen_features)} samples the model never sees")
    print(f"Input {tuple(features.shape)} -> target {tuple(targets.shape)}\n")

    # The loss at each epoch along the way, for the training curve
    losses = train_single_batch(model, loss_fn, optimizer, features, targets)

    print(f"\nLoss went from {losses[0]:.4f} to {losses[-1]:.6f}")
    print(f"Best loss was {min(losses):.6f} at epoch {losses.index(min(losses)) + 1}")

    plot_losses(losses)

    # Both losses and train_loss measure the finished model, but they differ in the data from that last update.
    train_loss = evaluate(model, loss_fn, features, targets)
    unseen_loss = evaluate(model, loss_fn, unseen_features, unseen_targets)

    print("\nRelative L2 loss after training")
    print(f"  on the training batch: {train_loss:.4f}")
    print(f"  on unseen data:        {unseen_loss:.4f}")
    print(f"  the model is {unseen_loss / train_loss:.1f}x worse on data it never saw")

    plot_generalisation(model, (features, targets), (unseen_features, unseen_targets))

    # Keep the trained weights, so compare_operators.py reuses this exact model
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")

    # The summary numbers and both figures, attached to this run
    wandb.log(
        {
            "final/train_loss": train_loss,
            "final/unseen_loss": unseen_loss,
            "final/generalisation_gap": unseen_loss / train_loss,
            "final/best_loss": min(losses),
            "loss_curve": wandb.Image(str(LOSS_PLOT_PATH)),
            "generalisation": wandb.Image(str(GENERALISATION_PLOT_PATH)),
        }
    )
    wandb.finish()
