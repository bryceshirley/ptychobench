from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.samples import (
    Apoferritin  # Import a Sample
)
from ptychobench.operators import (
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
    YevickThomsonOperator
)  # Import the Operators
from ptychobench.benchmark import run_benchmark  # The Physics Engine

# Grid parameters
grid_params = {
    "divergence_angle": 4.0, # Divergence angle of the initial Gaussian beam
    "lam": 1.0,  # Wavelength of electrons in nm (corresponding to ~300 keV)
    "L": 12.8,  # nm Total transverse width of the simulation domain
    "z_prop": 100.0,  # nm Total propagation distance
    "N": 400,  # Number of transverse pixels
    "Nz": 100,  # Number of longitudinal steps (z-steps)
    "probe_width": 1  # Width of the initial Gaussian beam
}

# 1. Setup Grid
grid = SimulationGrid(**grid_params)

# 2. Setup Sample 
sample = Apoferritin(0.001)

# 3. Operators to Benchmark
operators = [ParaxialOperator, FeitFleckOperator, LinDudaOperator, YevickThomsonOperator]


# 5. Run Benchmark
result = run_benchmark(grid, sample, operators)

# 6. Generate Plots and Save Parameters
result.generate_report()

# 7. Print a formatted summary to your console
result.print_summary()
