from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.samples import (
    Apoferritin,  # Import a Sample
    StraightWaveguides,
    SharpStraightWaveguides,
)
from ptychobench.operators import (
    ExactOperator,
    FeitFleckOperator,
)  # Import the Operators

import numpy as np
import torch


def generate_data():
    # 1. Create a define simulation parameters
    grid_params = {
        "divergence_angle": 4.0,  # Divergence angle of the initial Gaussian beam
        "lam": 0.00197,  # Wavelength of electrons in nm (corresponding to ~300 keV)
        "L": 12.8,  # nm Total transverse width of the simulation domain
        "z_prop": 100.0,  # nm Total propagation distance
        "N": 400,  # Number of transverse pixels
        "Nz": 2,  # Number of longitudinal steps (z-steps), kept small so dz = z_prop / Nz is large
        "probe_width": 4.0,  # Width of the initial Gaussian beam
    }

    # 2. Create a simulation grid
    grid = SimulationGrid(**grid_params)

    # Create an instance of the ExactOperator, for the ground truth
    exact_operator = ExactOperator(grid)

    # Create an instance of the FeitFleckOperator, for the classical baseline
    feit_fleck_operator = FeitFleckOperator(grid)

    # Define inputs and targets for the model
    num_configs = 2  # How many times to draw a fresh set of random parameters

    num_sample_types = 3  # Apoferritin, StraightWaveguides, SharpStraightWaveguides

    num_examples = (
        num_configs * num_sample_types * grid.Nz
    )  # configs * sample types * z-steps
    input_eps = torch.zeros((num_examples, grid.N), dtype=torch.complex64)
    input_psi = torch.zeros((num_examples, grid.N), dtype=torch.complex64)
    target_psi = torch.zeros((num_examples, grid.N), dtype=torch.complex64)
    baseline_psi = torch.zeros((num_examples, grid.N), dtype=torch.complex64)

    # Record what each row is, so nothing downstream has to recompute it from the loop order
    sample_names = []  # Which sample type the row came from
    config_ids = []  # Which draw of random parameters
    z_indices = []  # Which z-step, as an index
    z_values = []  # Which z-step, as a position in nm

    # Create random parameters for each sample type to generate diverse test cases.
    # The modulus spans an order of magnitude, so the dataset covers both the weak
    # perturbations where Feit/Fleck is accurate and the strong ones where it fails
    random_modulus_A = (
        0.01 + torch.rand(num_configs) * 0.09
    )  # Apoferritin's only tunable parameter, its geometry is fixed by the .npy file

    random_modulus_SW = (
        0.01 + torch.rand(num_configs) * 0.09
    )  # Strength of the StraightWaveguides' permittivity
    random_num_cores_SW = torch.randint(
        2, 11, (num_configs,)
    ).float()  # Number of waveguides across the domain (the spatial frequency)
    random_core_frac_SW = (
        0.1 + torch.rand(num_configs) * 0.3
    )  # Width of each waveguide, as a fraction of the period

    random_modulus_SSW = (
        0.01 + torch.rand(num_configs) * 0.09
    )  # Strength of the SharpStraightWaveguides' permittivity
    random_num_cores_SSW = torch.randint(
        2, 11, (num_configs,)
    ).float()  # Number of waveguides across the domain (the spatial frequency)
    random_core_frac_SSW = (
        0.2 + torch.rand(num_configs) * 0.4
    )  # Width of each waveguide, as a fraction of the period
    random_blur_SSW = (
        torch.rand(num_configs) * 4.0
    )  # Gaussian softening of the sharp edges, in pixels

    idx = 0
    for i in range(num_configs):
        # 3. Work out the waveguide spacing, so the core width can be set relative to it
        period_SW = (
            grid.L / random_num_cores_SW[i].item()
        )  # Spacing between the StraightWaveguides
        period_SSW = (
            grid.L / random_num_cores_SSW[i].item()
        )  # Spacing between the SharpStraightWaveguides

        # 4. Create a sample of each type, each with its own random parameters
        samples = [
            Apoferritin(
                modulus=random_modulus_A[i].item()
            ),  # Real data, only the modulus varies
            StraightWaveguides(  # Smooth Gaussian cores
                modulus=random_modulus_SW[i].item(),
                num_cores=random_num_cores_SW[i].item(),
                core_width=period_SW * random_core_frac_SW[i].item(),
            ),
            SharpStraightWaveguides(  # Square cores with optionally blurred edges
                modulus=random_modulus_SSW[i].item(),
                num_cores=random_num_cores_SSW[i].item(),
                core_width=period_SSW * random_core_frac_SSW[i].item(),
                blur=random_blur_SSW[i].item(),
            ),
        ]

        for sample in samples:
            psi = (
                grid.get_initial_field()
            )  # Initial wavefunction, restarted at z = 0 for each new sample

            # 5. Get the sample's potential (epsilon) and the current wavefunction (psi)
            for z_index, z in enumerate(grid.z_steps):
                eps = sample.get_permittivity(grid, z)  # Sample's potential at this z
                E = np.diag(
                    eps
                )  # Turn epsilon into the environment matrix the operators expect

                # 6. Store the input epsilon and psi
                input_eps[idx] = torch.from_numpy(eps).to(torch.complex64)
                input_psi[idx] = torch.from_numpy(psi).to(torch.complex64)

                # 7. Compute the ground truth wavefunction using the ExactOperator
                psi_exact = exact_operator.step(E, psi)

                # 8. Compute the classical baseline using the FeitFleckOperator, stepped from
                #    the same psi so the residual (exact - baseline) is well defined
                psi_feit_fleck = feit_fleck_operator.step(E, psi)

                # 9. Store the target psi and the classical baseline
                target_psi[idx] = torch.from_numpy(psi_exact).to(torch.complex64)
                baseline_psi[idx] = torch.from_numpy(psi_feit_fleck).to(torch.complex64)

                # 10. Label the row, so it can be found later without knowing the loop order
                sample_names.append(type(sample).__name__)
                config_ids.append(i)
                z_indices.append(z_index)
                z_values.append(float(z))

                # 11. Step the beam forward with the exact solution, so the next z-step
                #     starts from the true psi(z + dz)
                psi = psi_exact
                idx += 1

    # Loop that generates targets for the model
    data_dict = {
        "grid_params": grid_params,
        "input_eps": input_eps,
        "input_psi": input_psi,
        "target_psi": target_psi,
        "baseline_psi": baseline_psi,
        "sample_names": sample_names,
        "config_ids": config_ids,
        "z_indices": z_indices,
        "z_values": z_values,
    }
    torch.save(data_dict, "simulation_data.pt")


if __name__ == "__main__":
    generate_data()
