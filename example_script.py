# scripts.py (or your main runner file)

from ptychobench.grid import SimulationGrid
from ptychobench.samples import zig_balls
from ptychobench.operators import (
    ExactOperator,
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
)
from ptychobench.solver import run_benchmark
from ptychobench.utils import generate_results


def run_standard_simulation():
    grid = SimulationGrid(divergence_angle=20.0)
    sample_params = {"modulus": 0.75}

    operators = [ExactOperator, ParaxialOperator, FeitFleckOperator, LinDudaOperator]

    result = run_benchmark(
        grid=grid, sample=zig_balls, operators=operators, sample_params=sample_params
    )

    generate_results(grid, result)


if __name__ == "__main__":
    run_standard_simulation()
