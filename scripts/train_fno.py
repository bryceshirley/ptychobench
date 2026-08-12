"""
Step 3: train a basic FNO on the simulation data.

Run generate_data.py first, so that simulation_data.pt exists.
"""

from neuralop.models import FNO

from dataset import PtychoDataset

# The channel counts come straight from what PtychoDataset serves:
#   input  [Real(psi), Imag(psi), eps]        -> 3 channels
#   target [Real(psi_exact), Imag(psi_exact)] -> 2 channels
IN_CHANNELS = 3
OUT_CHANNELS = 2

# One entry per spatial dimension, and the wavefields are 1D across x
N_MODES = (16,)  # Number of Fourier modes the operator keeps
HIDDEN_CHANNELS = 64  # Width of the internal feature representation


def build_model():
    """Creates a basic FNO for the 1D wavefield data."""
    return FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
    )


if __name__ == "__main__":
    model = build_model()
    print(model)

    # Push one sample through, to check the model is wired to the data correctly
    dataset = PtychoDataset("simulation_data.pt")
    features, targets = dataset[0]
    features, targets = features.unsqueeze(0), targets.unsqueeze(0)  # Add a batch dimension

    prediction = model(features)
    print(f"\nInput shape:      {tuple(features.shape)}")
    print(f"Prediction shape: {tuple(prediction.shape)}")
    print(f"Target shape:     {tuple(targets.shape)}")
