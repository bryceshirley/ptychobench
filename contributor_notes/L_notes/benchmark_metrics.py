from ptychobench.metrics import calculate_rmse
import time
import numpy as np


def benchmark_metrics_python():
    njit_time = 0
    no_njit_time = 0
    COUNT = 100000
    x = np.random.randint(0, 1000, COUNT)
    y = np.random.randint(0, 1000, COUNT)

    N = 10

    calculate_rmse(x, y)
    calculate_rmse_no_njit(x, y)
    for _ in range(N):
        start = time.time()
        calculate_rmse(x, y)
        benchmark = time.time() - start
        print(benchmark)
        njit_time += benchmark

        start = time.time()
        calculate_rmse_no_njit(x, y)
        benchmark = time.time() - start
        print(benchmark)
        no_njit_time += benchmark

        print()

    njit_time /= N
    no_njit_time /= N
    print(f"NJIT Time:\t{njit_time}")
    print(f"No NJIT Time:\t{no_njit_time}")


def calculate_rmse_no_njit(exact, approx) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between two arrays.

    Formula: RMSE = sqrt( (1/N) * sum( |exact_i - approx_i|^2 ) )

    Parameters:
        exact (np.ndarray): The ground truth data array.
        approx (np.ndarray): The approximated data array.

    Returns:
        float: The calculated error.
    """
    # Get the number of elements in the arrays
    N = len(exact)
    # N = exact.__len__()

    # TODO: Students to implement this using Ping-Pong TDD!
    # Hints:
    #  - Use a standard 'for' loop to represent the summation formula.
    #  - Use the built-in abs() function to compute the absolute value ie |exact_i - approx_i|
    # return 1.0  # Placeholder return value
    sum = 0
    for idx in range(N):
        x = exact[idx]
        y = approx[idx]
        sum += abs(x - y) ** 2
    sum /= N

    return np.sqrt(sum)


# if __name__ == "main":
benchmark_metrics_python()
