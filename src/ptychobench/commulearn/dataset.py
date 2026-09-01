"""
PyTorch Dataset for the simulation data written by generate_data.py.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATA_PATH = Path(
    "scripts/data/simulation_data.pt"
)  # Where generate_data.py writes the dataset


class PtychoDataset(Dataset):
    def __init__(self, path=DATA_PATH):
        data = torch.load(path)
        self.psi_in = data["input_psi"]
        self.eps_in = data["input_eps"]
        self.psi_target = data["target_psi"]
        assert len(self.psi_in) == len(self.eps_in) == len(self.psi_target)

    def __len__(self):
        return len(self.psi_in)

    def __getitem__(self, index):
        psi, eps = self.psi_in[index], self.eps_in[index]
        psi_target = self.psi_target[index]
        input_stacked = torch.stack([psi.real, psi.imag, eps.real])
        output_stacked = torch.stack([psi_target.real, psi_target.imag])
        return input_stacked, output_stacked


if __name__ == "__main__":
    training_data = PtychoDataset()
    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

    train_features, train_targets = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Target batch shape: {train_targets.size()}")
