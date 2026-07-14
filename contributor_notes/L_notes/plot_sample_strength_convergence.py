from ptychobench.grid import SimulationGrid
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
)
from ptychobench.benchmark import run_benchmark
from matplotlib import pyplot as plt
import numpy as np

NEW_TESTS = True
# NEW_TESTS = False

moduli_1 = [0.1, 0.5, 1.0, 1.5, 2.0]
moduli_2 = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
moduli = moduli_2

tests_1 = [
    (0.1, moduli[0], None, None, 0),
    (0.1, moduli[1], None, None, 0),
    (0.1, moduli[2], None, None, 0),
    (0.1, moduli[3], None, None, 0),
    (0.1, moduli[4], None, None, 0),
]
tests_2 = [(0.1, modulus, None, None, 0) for modulus in moduli]
tests_3 = [(0, modulus, None, None, 0) for modulus in moduli]
tests_4 = [(40, modulus, None, None, 0) for modulus in moduli]
tests = tests_4
SAMPLES = [
    Apoferritin,
    StraightWaveguides,
    ZigWaveguides,
    StraightBalls,
    ZigBalls,
]

FILE_1 = "scripts/script_data/errors_exact.npy"
FILE_2 = "scripts/script_data/errors_paraxial.npy"
FILE_3 = "scripts/script_data/errors_feit_fleck.npy"
FILE_4 = "scripts/script_data/errors_lin_duda.npy"
errors_exact = []
errors_paraxial = []
errors_feit_fleck = []
errors_lin_duda = []


def do_tests():
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

        result = run_benchmark(grid, sample, operators)
        # errors.append(result.errors["ExactOperator"])
        print(result)
        errors_exact.append(result.errors["ExactOperator"])
        errors_paraxial.append(result.errors["ParaxialOperator"])
        errors_feit_fleck.append(result.errors["FeitFleckOperator"])
        errors_lin_duda.append(result.errors["LinDudaOperator"])


if NEW_TESTS:
    do_tests()
    np.save(arr=errors_exact, file=FILE_1)
    np.save(arr=errors_paraxial, file=FILE_2)
    np.save(arr=errors_feit_fleck, file=FILE_3)
    np.save(arr=errors_lin_duda, file=FILE_4)
else:
    try:
        errors_exact = np.load(FILE_1)
        errors_paraxial = np.load(FILE_2)
        errors_feit_fleck = np.load(FILE_3)
        errors_lin_duda = np.load(FILE_4)
    except OSError:
        do_tests()

# print(errors_exact)
# print(errors_paraxial)
# print(errors_feit_fleck)
# print(errors_lin_duda)

fig, axes = plt.subplots()
plt.plot(moduli, errors_paraxial, "-o", c="red", label="Paraxial Operator")
plt.plot(moduli, errors_feit_fleck, "-o", c="yellow", label="Feit/Fleck Operator")
plt.plot(moduli, errors_lin_duda, "-o", c="green", label="Lin/Duda Operator")
axes.set_xlabel("Modulus")
axes.set_ylabel("RMSE")
plt.legend()
plt.show()
