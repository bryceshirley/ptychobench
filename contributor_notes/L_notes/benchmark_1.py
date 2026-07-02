from typing import List

from ptychobench.grid import SimulationGrid  # Simulation Sandbox
from ptychobench.results import BenchmarkResult
from ptychobench.samples import (
    Sample,
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
import numpy as np


def plot_wavefields_grid(grid, wavefield_history):
    """Plots the propagated wavefields (phase and amplitude) for each operator."""
    extent = [-grid.L / 2, grid.L / 2, grid.z_prop, 0]

    n_ops = len(wavefield_history)
    col_count = n_ops // 4
    fig_p, axes_p = plt.subplots(4, col_count, figsize=(6 * n_ops, 6), sharey=True)
    fig_a, axes_a = plt.subplots(4, col_count, figsize=(6 * n_ops, 6), sharey=True)

    if n_ops == 1:
        axes_p, axes_a = [axes_p], [axes_a]

    print(axes_p)
    j = 0
    for i, (name, data) in enumerate(wavefield_history):
        i = i // col_count
        im_p = axes_p[i, j].imshow(
            np.angle(data), extent=extent, aspect="auto", cmap="magma"
        )
        axes_p[i, j].set_title(name, fontsize=14)
        axes_p[i, j].set_xlabel("Transverse coordinate x", fontsize=12)

        im_a = axes_a[i, j].imshow(
            np.abs(data), extent=extent, aspect="auto", cmap="magma"
        )
        axes_a[i, j].set_title(name, fontsize=14)
        axes_a[i, j].set_xlabel("Transverse coordinate x", fontsize=12)
        j += 1
        j %= col_count

    axes_p[0, 0].set_ylabel("Propagation Distance z", fontsize=12)
    fig_p.colorbar(im_p, ax=axes_p, fraction=0.02, pad=0.02)
    fig_p.suptitle("2D Field Propagation (Phase)", fontsize=16)

    axes_a[0, 0].set_ylabel("Propagation Distance z", fontsize=12)
    fig_a.colorbar(im_a, ax=axes_a, fraction=0.02, pad=0.02)
    fig_a.suptitle("2D Field Propagation (Amplitude)", fontsize=16)

    return fig_p, fig_a


# fig_p, axes_p = plt.subplots(4, 2, figsize=(6 * 8, 6), sharey=True)
# print(axes_p)
# print(axes_p[0])
# print(axes_p[0,0])

# DIVERGENCE_ANGLE = 0
# MODULUS = .5
# SAMPLES = Sample.__subclasses__()
SAMPLES = [
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
]
# SAMPLE = SAMPLES[0]
tests = [
    (20.0, 0.5, None, None, 0),
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
tests_1 = tests[0]
tests_2 = tests[1:5]
tests_3 = tests[5:9]
tests_4 = tests[9:13]
tests_5 = tests[13:17]
tests_6 = tests[17:21]
tests_7 = tests[21:25]

errors = []
results: List[BenchmarkResult] = []
wavefield_history = []
tests = tests_7
for test in tests:
    # for test in tests[21:23]:
    print(test)
    divergence_angle = test[0]
    modulus = test[1]
    period = test[2]
    core_width = test[3]
    sample: Sample = SAMPLES[test[4]]

    grid = SimulationGrid(divergence_angle=divergence_angle)

    if period is not None and core_width is not None:
        sample = sample(modulus=modulus, period=period, core_width=core_width)
    elif period is not None:
        sample = sample(modulus=modulus, period=period)
    elif core_width is not None:
        sample = sample(modulus=modulus, core_width=core_width)
    else:
        sample = sample(modulus=modulus)
    operators = [ParaxialOperator, FeitFleckOperator, LinDudaOperator]
    # sample.plot(grid)

    result = run_benchmark(grid, sample, operators)
    # result.generate_report()
    # result.print_summary()

    # fig_p, fig_a = result.plot_wavefields()

    errors.append(result.errors)
    results.append(result)
    # wavefield_history.apend(result.wavefield_history)

# wavefield_history = results[0].wavefield_history
# for result in results[1:]:
#     for key, val in result.items():
#         wavefield_history
wavefield_history = []
col_count = len(tests)
for result in results:
    for i, (key, val) in enumerate(result.wavefield_history.items()):
        wavefield_history.insert(
            i % col_count + i // col_count, (f"{key} {len(wavefield_history)}", val)
        )
plot_wavefields_grid(results[0].grid, wavefield_history)
figure = plt.figure()
plt.subplots_adjust(left=0.13, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.5)
plt.show()

for i in errors:
    for key, value in i.items():
        if value != 0:
            print(f"{key:<12}: {value}")
    print()
