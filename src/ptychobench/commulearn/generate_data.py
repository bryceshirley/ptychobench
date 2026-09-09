"""
Generate the training dataset for the neural operator.

Simulates randomised samples with the classical operators and records, at every
propagation step, the model inputs, the exact target, and the Feit/Fleck baseline.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from ptychobench.commulearn.dataset import DATA_PATH
from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.operators import (  # Import the Operators
    FeitFleckOperator,
    GroundTruthOperator,
    Integrator,
)
from ptychobench.samples import (
    Apoferritin,  # Import a Sample
    Sample,
    SharpStraightWaveguides,
    StraightWaveguides,
)

#: The grid every sample is simulated on, unless generate_data is given another.
DEFAULT_GRID_PARAMS: dict[str, Any] = {
    "divergence_angle": 4.0,  # Divergence angle of the initial Gaussian beam
    "lam": 0.00197,  # Wavelength of electrons in nm (corresponding to ~300 keV)
    "L": 12.8,  # nm Total transverse width of the simulation domain
    "z_prop": 100.0,  # nm Total propagation distance
    "N": 400,  # Number of transverse pixels
    "Nz": 2,  # Number of longitudinal steps (z-steps), kept small so dz = z_prop / Nz is large
    "probe_width": 4.0,  # Width of the initial Gaussian beam
}


def to_host(array: Any) -> torch.Tensor:
    """Copy a field into a host complex64 tensor, whatever backend produced it.

    as_tensor rather than from_numpy, because a torch grid hands back tensors
    rather than numpy arrays, and .cpu() because those tensors may be on the GPU
    while the dataset buffers are not.
    """
    return torch.as_tensor(array).cpu().to(torch.complex64)


def _parameters_of(sample: Sample) -> dict[str, Any]:
    """
    Read back what a sample was built with, so it can be rebuilt from the dataset.

    Declared fields rather than ``vars()``, so the cached arrays some samples keep
    are left out of the record. Paths are stored as strings: the strict loader
    torch.load defaults to rejects Path objects, and the samples take either.
    """
    values = (
        {f.name: getattr(sample, f.name) for f in fields(sample)}
        if is_dataclass(sample)
        else vars(sample).copy()
    )
    return {k: str(v) if isinstance(v, Path) else v for k, v in values.items()}


def random_samples(grid: SimulationGrid) -> list[Sample]:
    """
    Draw one random sample of each type.

    The modulus spans an order of magnitude, so the dataset covers both the weak
    perturbations where Feit/Fleck is accurate and the strong ones where it fails.
    The core width is drawn as a fraction of the waveguide period rather than as an
    absolute width, so the guides stay separated however many cores come out.

    Parameters
    ----------
    grid : SimulationGrid
        The grid the samples are simulated on. Supplies the domain width the
        waveguide period is measured against.

    Returns
    -------
    list of Sample
        One sample of each type, each with its own random parameters.
    """
    num_cores_SW = torch.randint(2, 11, (1,)).item()  # The spatial frequency
    num_cores_SSW = torch.randint(2, 11, (1,)).item()

    period_SW = grid.L / num_cores_SW  # Spacing between the StraightWaveguides
    period_SSW = grid.L / num_cores_SSW

    return [
        Apoferritin(  # Real data, only the modulus varies
            modulus=(0.01 + torch.rand(1) * 0.09).item(),
        ),
        StraightWaveguides(  # Smooth Gaussian cores
            modulus=(0.01 + torch.rand(1) * 0.09).item(),
            num_cores=float(num_cores_SW),
            core_width=period_SW * (0.1 + torch.rand(1) * 0.3).item(),
        ),
        SharpStraightWaveguides(  # Square cores with optionally blurred edges
            modulus=(0.01 + torch.rand(1) * 0.09).item(),
            num_cores=float(num_cores_SSW),
            core_width=period_SSW * (0.2 + torch.rand(1) * 0.4).item(),
            blur=(
                torch.rand(1) * 4.0
            ).item(),  # Softening of the sharp edges, in pixels
        ),
    ]


def generate_data(
    grid_params: dict[str, Any] | None = None,
    num_configs: int = 2,
    sample_builder: Callable[[SimulationGrid], Sequence[Sample]] = random_samples,
    integrator: Integrator = "direct",
    path: Path = DATA_PATH,
) -> None:
    """
    Simulate randomised samples and save the training pairs.

    Walks through randomised sample parameters and, at every propagation step,
    records the wavefield going in, the sample it passes through, the exact
    answer, and what the classical Feit/Fleck operator predicts.

    Parameters
    ----------
    grid_params : dict, optional
        The `SimulationGrid` settings shared by every sample. Defaults to
        :data:`DEFAULT_GRID_PARAMS`.
    num_configs : int, optional
        How many times to draw a fresh set of random sample parameters.
    sample_builder : callable, optional
        Draws the samples for one configuration. Defaults to
        :func:`random_samples`, which gives one of each of the three types. The
        number of sample types is taken from what this returns, so it never has
        to be kept in step by hand.
    integrator : str, optional
        How the operators evaluate the matrix exponential.
    path : pathlib.Path, optional
        Where to write the dataset. Defaults to
        :data:`~ptychobench.commulearn.dataset.DATA_PATH`.
    """
    # 1. Create a define simulation parameters
    grid_params = DEFAULT_GRID_PARAMS if grid_params is None else grid_params

    # 2. Create a simulation grid
    grid = SimulationGrid(
        divergence_angle=grid_params["divergence_angle"],
        lam=grid_params["lam"],
        L=grid_params["L"],
        z_prop=grid_params["z_prop"],
        N=grid_params["N"],
        Nz=grid_params["Nz"],
        probe_width=grid_params["probe_width"],
        backend="torch",
    )

    # Create an instance of the GroundTruthOperator, for the ground truth
    exact_operator = GroundTruthOperator(grid, integrator=integrator)

    # Create an instance of the FeitFleckOperator, for the classical baseline
    feit_fleck_operator = FeitFleckOperator(grid, integrator=integrator)

    # 3. Draw every configuration up front, so the buffers can be sized from what
    #    was actually built rather than from a count kept in step by hand
    configurations = [sample_builder(grid) for _ in range(num_configs)]
    num_sample_types = len(configurations[0])

    # Define inputs and targets for the model
    num_examples = (
        num_configs * num_sample_types * grid.Nz
    )  # configs * sample types * z-steps
    input_eps = torch.zeros((num_examples, grid.N), dtype=torch.complex64)
    input_psi = torch.zeros((num_examples, grid.N), dtype=torch.complex64)
    target_psi = torch.zeros((num_examples, grid.N), dtype=torch.complex64)
    baseline_psi = torch.zeros((num_examples, grid.N), dtype=torch.complex64)

    # Record what each row is, so nothing downstream has to recompute it from the loop order
    sample_names: list[str] = []  # Which sample type the row came from
    config_ids: list[int] = []  # Which draw of random parameters
    z_indices: list[int] = []  # Which z-step, as an index
    z_values: list[float] = []  # Which z-step, as a position in nm
    sample_params: list[dict[str, Any]] = []  # What the sample was built with

    idx = 0
    for i, samples in enumerate(configurations):
        for sample in samples:
            psi = (
                grid.get_initial_field()
            )  # Initial wavefunction, restarted at z = 0 for each new sample

            # 4. Get the sample's potential (epsilon) and the current wavefunction (psi)
            for z_index, z in enumerate(grid.z_steps):
                eps = sample.get_permittivity(grid, z)  # Sample's potential at this z

                # 5. Store the input epsilon and psi, on the host whatever backend
                #    the grid used
                input_eps[idx] = to_host(eps)
                input_psi[idx] = to_host(psi)

                # 6. Compute the ground truth wavefunction using the GroundTruthOperator.
                #    step() takes epsilon as a length-N vector or as diag(eps) and builds
                #    whichever it needs, so pass the vector and skip the N x N matrix
                psi_exact = exact_operator.step(eps, psi)

                # 7. Compute the classical baseline using the FeitFleckOperator, stepped from
                #    the same psi so the residual (exact - baseline) is well defined
                psi_feit_fleck = feit_fleck_operator.step(eps, psi)

                # 8. Store the target psi and the classical baseline
                target_psi[idx] = to_host(psi_exact)
                baseline_psi[idx] = to_host(psi_feit_fleck)

                # 9. Label the row, so it can be found later without knowing the loop order
                sample_names.append(type(sample).__name__)
                sample_params.append(_parameters_of(sample))
                config_ids.append(i)
                z_indices.append(z_index)
                z_values.append(float(z))

                # 10. Step the beam forward with the exact solution, so the next z-step
                #     starts from the true psi(z + dz)
                psi = psi_exact
                idx += 1

    # Bundle everything into one file, for the DataLoader to read back later
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
        "sample_params": sample_params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_dict, path)
    print(f"Saved {num_examples} rows to {path}")


if __name__ == "__main__":
    generate_data()
