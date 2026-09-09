"""
Step 3: train a basic FNO on the simulation data.

Run generate_data.py first, so that simulation_data.pt exists.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from neuralop.losses import LpLoss
from neuralop.models import FNO
from torch.utils.data import DataLoader, Subset, random_split

from ptychobench.commulearn.dataset import PtychoDataset


@dataclass
class TrainingConfig:
    """
    Everything a training run needs, in one place.

    Gathered into a dataclass so a run can be described by a value rather than by
    the state of the module: ``replace(CONFIG, n_modes=(16,))`` gives a variant
    without editing anything, and ``asdict`` hands the whole lot to Weights and
    Biases so what is logged cannot drift from what was used.

    Attributes
    ----------
    in_channels, out_channels : int
        Set by what `PtychoDataset` serves, so leave them alone unless the
        channel layout changes:
        input ``[Real(psi), Imag(psi), eps]`` and target
        ``[Real(psi_exact), Imag(psi_exact)]``.
    n_modes : tuple of int
        Fourier modes the operator keeps, one entry per spatial dimension. The
        wavefields are 1D, so this is a one-element tuple. The FNO discards
        everything above it, so too few caps how well it can ever fit.
    hidden_channels : int
        Width of the internal feature representation.
    learning_rate : float
        The learning rate for Adam.
    batch_size : int
        Size of the single batch the model trains on. Also the size of the
        training split, since that batch is the whole of it.
    num_epochs : int
        How many passes over that one batch.
    split_seed : int
        Fixes which rows are held back, so "unseen" means the same thing on every
        run and across scripts.
    """

    # The channel counts come straight from what PtychoDataset serves
    in_channels: int = 3
    out_channels: int = 2

    # One entry per spatial dimension, and the wavefields are 1D across x
    n_modes: tuple[int, ...] = (64,)
    hidden_channels: int = 64  # Width of the internal feature representation

    learning_rate: float = 1e-3  # The usual starting point for an FNO
    batch_size: int = 8  # The single batch the model is trained on
    num_epochs: int = 5000  # Many passes over that one batch, so it can memorise it

    split_seed: int = 0  # Fixes which samples are held out
    wandb_project: str = "ptychobench-fno"  # Every run shows up under this project

    loss_plot_path: Path = Path("scripts/results/training_loss.png")
    generalisation_plot_path: Path = Path("scripts/results/generalisation.png")
    # compare_operators.py loads the model from here
    weights_path: Path = Path("scripts/results/fno_weights.pt")

    def __post_init__(self) -> None:
        """Catch the settings that would otherwise fail deep inside a training run."""
        if isinstance(self.n_modes, int):
            # (16) is an int, not a tuple: the trailing comma is easy to leave out
            raise TypeError("n_modes must be a tuple, e.g. (64,) rather than (64)")
        if not self.n_modes or any(m < 1 for m in self.n_modes):
            raise ValueError(f"n_modes must be positive, got {self.n_modes}")
        for name in ("in_channels", "out_channels", "hidden_channels", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )
        if self.num_epochs < 10:
            # The progress line prints every num_epochs // 10 epochs
            raise ValueError(f"num_epochs must be at least 10, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )


#: The settings the script runs with. Pass another to any function below to vary it.
CONFIG = TrainingConfig()


def build_model(config: TrainingConfig = CONFIG) -> FNO:
    """1. Creates a basic FNO for the 1D wavefield data."""
    return FNO(
        n_modes=config.n_modes,
        hidden_channels=config.hidden_channels,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
    )


def build_loss() -> LpLoss:
    """
    2. Relative L2 loss, the standard choice for Fourier neural operators.

    Being relative, it normalises by the size of the target, so weak and strong
    perturbations count equally. reduction="mean" keeps the value independent of
    the batch size, and 0.1 reads as a 10% error.
    """
    return LpLoss(d=1, p=2, reduction="mean")


def build_optimizer(model: FNO, config: TrainingConfig = CONFIG) -> torch.optim.Adam:
    """2. Adam, as suggested in the issue."""
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def train_single_batch(
    model: FNO,
    loss_fn: LpLoss,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig = CONFIG,
) -> list[float]:
    """
    3. Trains on one fixed batch, giving the model every chance to memorise it.

    Returns the history of losses seen during training, one per epoch. Each entry is
    measured just before that epoch's update, so it describes the model as it was
    partway through training, not the model this function hands back. Use it to draw
    the training curve, and evaluate() to measure the finished model.
    """
    model.train()
    losses: list[float] = []

    for epoch in range(config.num_epochs):
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

        if (epoch + 1) % (config.num_epochs // 10) == 0:
            print(
                f"  epoch {epoch + 1:>5}/{config.num_epochs}   loss {loss.item():.6f}"
            )

    return losses


def plot_losses(losses: list[float], config: TrainingConfig = CONFIG) -> None:
    """4. Plots the loss against the epoch, on a log scale as it spans two orders of magnitude."""
    config.loss_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 loss")
    ax.set_title(f"Training loss on a single batch of {config.batch_size} samples")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(config.loss_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve to {config.loss_plot_path}")


def evaluate(
    model: FNO, loss_fn: LpLoss, features: torch.Tensor, targets: torch.Tensor
) -> float:
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
    model: FNO,
    train_batch: tuple[torch.Tensor, torch.Tensor],
    unseen_batch: tuple[torch.Tensor, torch.Tensor],
    config: TrainingConfig = CONFIG,
) -> None:
    """5. Compares predictions against the truth, for a trained sample and an unseen one."""
    config.generalisation_plot_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(config.generalisation_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved generalisation plot to {config.generalisation_plot_path}")


def split_dataset(
    dataset: PtychoDataset, config: TrainingConfig = CONFIG
) -> list[Subset]:
    """
    Hold some rows back, so there is something the model has genuinely never seen.

    The seed lives in the config, so compare_operators.py can reproduce exactly
    this split and label its figures with it.
    """
    generator = torch.Generator().manual_seed(config.split_seed)
    return random_split(
        dataset,
        [config.batch_size, len(dataset) - config.batch_size],
        generator=generator,
    )


if __name__ == "__main__":
    config = CONFIG

    # Record the hyperparameters alongside the results, so runs can be compared later.
    # asdict means a new field is logged without anyone remembering to add it here
    wandb.init(project=config.wandb_project, config=asdict(config))

    model = build_model(config)
    loss_fn = build_loss()
    optimizer = build_optimizer(model, config)

    dataset = PtychoDataset()
    train_set, unseen_set = split_dataset(dataset, config)

    # One fixed batch to train on, and one batch of everything held back
    features, targets = next(iter(DataLoader(train_set, batch_size=config.batch_size)))
    unseen_features, unseen_targets = next(
        iter(DataLoader(unseen_set, batch_size=len(unseen_set)))
    )

    print(
        f"Training on a single batch of {len(features)} samples "
        f"for {config.num_epochs} epochs"
    )
    print(f"Holding back {len(unseen_features)} samples the model never sees")
    print(f"Input {tuple(features.shape)} -> target {tuple(targets.shape)}\n")

    # The loss at each epoch along the way, for the training curve
    losses = train_single_batch(model, loss_fn, optimizer, features, targets, config)

    print(f"\nLoss went from {losses[0]:.4f} to {losses[-1]:.6f}")
    print(f"Best loss was {min(losses):.6f} at epoch {losses.index(min(losses)) + 1}")

    plot_losses(losses, config)

    # Both scores below measure the finished model, so they differ only in the data.
    # train_loss will not quite match losses[-1], measured one update earlier.
    train_loss = evaluate(model, loss_fn, features, targets)
    unseen_loss = evaluate(model, loss_fn, unseen_features, unseen_targets)

    print("\nRelative L2 loss after training")
    print(f"  on the training batch: {train_loss:.4f}")
    print(f"  on unseen data:        {unseen_loss:.4f}")
    print(f"  the model is {unseen_loss / train_loss:.1f}x worse on data it never saw")

    plot_generalisation(
        model, (features, targets), (unseen_features, unseen_targets), config
    )

    # Keep the trained weights, so compare_operators.py reuses this exact model
    torch.save(model.state_dict(), config.weights_path)
    print(f"Saved weights to {config.weights_path}")

    # The summary numbers and both figures, attached to this run
    wandb.log(
        {
            "final/train_loss": train_loss,
            "final/unseen_loss": unseen_loss,
            "final/generalisation_gap": unseen_loss / train_loss,
            "final/best_loss": min(losses),
            "loss_curve": wandb.Image(str(config.loss_plot_path)),
            "generalisation": wandb.Image(str(config.generalisation_plot_path)),
        }
    )
    wandb.finish()
