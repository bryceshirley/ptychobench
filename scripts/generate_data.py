from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.samples import (
    Apoferritin,  # Import a Sample
    StraightWaveguides,
    SharpStraightWaveguides
)
from ptychobench.operators import (
    ExactOperator
)  # Import the Operators

import torch
from torch.utils.data import Dataset

def generate_data():
    # 1. Create a define simulation parameters
    grid_params = {
        "divergence_angle": 4.0, # Divergence angle of the initial Gaussian beam
        "lam": 0.00197,  # Wavelength of electrons in nm (corresponding to ~300 keV)
        "L": 12.8,  # nm Total transverse width of the simulation domain
        "z_prop": 40.0,  # nm Total propagation distance
        "N": 400,  # Number of transverse pixels
        "Nz": 2,  # Number of longitudinal steps (z-steps)
        "probe_width": 4.0  # Width of the initial Gaussian beam
    }

    # 2. Create a simulation grid
    grid = SimulationGrid(**grid_params)

    # Create an instance of the ExactOperator
    operator = ExactOperator(grid)

    # Define inputs and targets for the model
    total_samples = 2 # Total number of test cases

    total_tests = total_samples * grid.Nz  # Total number of tests (samples * z-steps)
    input_eps = torch.zeros((total_tests, grid.N), dtype=torch.complex64)
    input_psi = torch.zeros((total_tests, grid.N), dtype=torch.complex64)
    target_psi = torch.zeros((total_tests, grid.N), dtype=torch.complex64)

    # Create a random modulus for the sample's permittivity to generate diverse test cases
    random_modulus = torch.rand(total_samples)*0.01

    idx = 0
    for i in range(total_samples):
        # 3. Create a sample (e.g., Apoferritin)
        sample = Apoferritin(modulus=random_modulus[i].item())  # Create a sample with random modulus

        # 4. Get the sample's potential (epsilon) and initial wavefunction (psi)
        for j in range(grid.Nz):
            eps = sample.get_permittivity(grid, j)  # Sample's potential
            psi = grid.get_initial_field()  # Initial wavefunction

            # 5. Store the input epsilon and psi
            input_eps[idx] = torch.from_numpy(eps).to(torch.complex64)
            input_psi[idx] = torch.from_numpy(psi).to(torch.complex64)

            # 6. Compute the target wavefunction using the ExactOperator
            target_psi_i = operator.step(psi, eps)  # Apply the operator to get the target psi

            # 7. Store the target psi
            target_psi[idx] = torch.from_numpy(target_psi_i).to(torch.complex64)
            idx += 1

    # Loop that generates targets for the model
    data_dict = {
                "grid_params": grid_params,
                "input_eps": input_eps,
                "input_psi": input_psi,
                "target_psi": target_psi
        }
    torch.save(data_dict, "simulation_data.pt")


class PtychoDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.psi_in = data['input_eps']
        self.eps_in = data['input_psi']
        self.psi_target = data['target_psi']
        
    def __len__(self): return len(self.psi_in)
    def __getitem__(self, idx): return self.psi_in[idx], self.eps_in[idx], self.psi_target[idx]


if __name__ == "__main__":
    generate_data()