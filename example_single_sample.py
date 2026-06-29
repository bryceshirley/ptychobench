from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.samples import Apoferritin  # Import a Sample
from ptychobench.operators import (
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
)  # Import the Operators
from ptychobench.benchmark import run_benchmark  # The Physics Engine

# 1. Setup Grid (Set the beam spread to 20 degrees)
grid = SimulationGrid(divergence_angle=20)

# 2. Setup Sample (Choose Apoferritin with a strength of 0.8)
sample = Apoferritin(modulus=0.8)

# 3. Operators to Benchmark (The mathematical solvers we are testing)
operators = [ParaxialOperator, FeitFleckOperator, LinDudaOperator]

# 4. Run Benchmark
result = run_benchmark(grid, sample, operators)

# 5. Generate Plots and Save Results
result.generate_report()
