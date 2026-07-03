from typing import List

from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.results import BenchmarkResult
from ptychobench.samples import (
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
)
from ptychobench.operators import (
    ParaxialOperator,
    FeitFleckOperator,
    LinDudaOperator,
)  # Import the Operators
from ptychobench.benchmark import run_benchmark  # The Physics Engine
from matplotlib import pyplot as plt

SAMPLES = [
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
]
# Each tuple is a test with the format:
# (divergence_angle, modulus, period, core_width, sample_idx)
# period and core_width may be None for samples that do not have them as variables
tests = [
    (80.0, 0.01, 10.0, 1.5, 1),
    # [1:5] 1. Changing modulus
    (0.0, 0.2, None, None, 0),
    (0.0, 0.4, None, None, 0),
    (0.0, 0.6, None, None, 0),
    (0.0, 0.8, None, None, 0),
    # [5:9] 2. Changing divergence
    (0.0, 0.5, None, None, 0),
    (5.0, 0.5, None, None, 0),
    (10.0, 0.5, None, None, 0),
    (15.0, 0.5, None, None, 0),
    # [9:13] 3. Changing modulus
    (0.0, 1.0, 10.0, 1.5, 1),
    (0.0, 2.0, 10.0, 1.5, 1),
    (0.0, 3.0, 10.0, 1.5, 1),
    (0.0, 4.0, 10.0, 1.5, 1),
    # [13:17] 3. Changing divergence
    (0.0, 1.0, 10.0, 1.5, 1),
    (10.0, 1.0, 10.0, 1.5, 1),
    (20.0, 1.0, 10.0, 1.5, 1),
    (30.0, 1.0, 10.0, 1.5, 1),
    # [17:21] 4. Changing period
    (10.0, 1.0, 10.0, 1.5, 1),
    (10.0, 1.0, 20.0, 1.5, 1),
    (10.0, 1.0, 30.0, 1.5, 1),
    (10.0, 1.0, 40.0, 1.5, 1),
    # [21:25] 4. Changing core width
    (10.0, 1.0, 10.0, 2.0, 1),
    (10.0, 1.0, 10.0, 4.0, 1),
    (10.0, 1.0, 10.0, 6.0, 1),
    (10.0, 1.0, 10.0, 8.0, 1),
]
tests_1 = tests[0:1]
tests_2 = tests[1:5]
tests_3 = tests[5:9]
tests_4 = tests[9:13]
tests_5 = tests[13:17]
tests_6 = tests[17:21]
tests_7 = tests[21:25]

errors = []
results: List[BenchmarkResult] = []
wavefield_history = []
tests = tests_1
for test in tests:
    # for test in tests[21:23]:
    print(test)
    divergence_angle = test[0]
    modulus = test[1]
    period = test[2]
    core_width = test[3]
    sample = SAMPLES[test[4]]

    grid = SimulationGrid(divergence_angle=divergence_angle)

    if period is not None and core_width is not None:
        sample = sample(
            modulus=modulus,
            period=period,  # ty:ignore[unknown-argument]
            core_width=core_width,  # ty:ignore[unknown-argument]
        )
    elif period is not None:
        sample = sample(
            modulus=modulus,
            period=period,  # ty:ignore[unknown-argument]
        )
    elif core_width is not None:
        sample = sample(
            modulus=modulus,
            core_width=core_width,  # ty:ignore[unknown-argument]
        )
    else:
        sample = sample(modulus=modulus)
    operators = [ParaxialOperator, FeitFleckOperator, LinDudaOperator]
    # sample.plot(grid)

    result = run_benchmark(grid, sample, operators)
    result.generate_report()
    result.print_summary()

    fig_p = result.plot_farfield_intensity_error()

    errors.append(result.errors)
    results.append(result)

plt.show()
