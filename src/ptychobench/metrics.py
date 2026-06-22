import numpy as np
from numba import njit


@njit
def calculate_rmse(exact: np.ndarray, approx: np.ndarray) -> float:
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
    # N = exact.size

    # TODO: Students to implement this using Ping-Pong TDD!
    # Hints:
    #  - Use a standard 'for' loop to represent the summation formula.
    #  - Use the built-in abs() function to compute the absolute value ie |exact_i - approx_i|
    pass
