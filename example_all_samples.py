from ptychobench.grid import SimulationGrid
from ptychobench.samples import (
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
)
from ptychobench.operators import ParaxialOperator, FeitFleckOperator, LinDudaOperator
from ptychobench.benchmark import run_benchmark


# 1. Setup Grid
grid = SimulationGrid(divergence_angle=20.0)

# 3. Operators to Benchmark
operators = [ParaxialOperator, FeitFleckOperator, LinDudaOperator]

for sample_class in [
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
]:
    sample = sample_class(modulus=0.8)

    # 4. Run Benchmark
    result = run_benchmark(grid, sample, operators)

    # 5. Generate Plots and Save Results
    result.generate_plots()
