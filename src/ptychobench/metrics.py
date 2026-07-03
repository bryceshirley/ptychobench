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
    N = len(exact)
    # N = exact.__len__()

    # TODO: Students to implement this using Ping-Pong TDD!
    # Hints:
    #  - Use a standard 'for' loop to represent the summation formula.
    #  - Use the built-in abs() function to compute the absolute value ie |exact_i - approx_i|
    # return 1.0  # Placeholder return value
    total = 0
    for idx in range(N):
        x = exact[idx]
        y = approx[idx]
        total += abs(x - y) ** 2
    total /= N

    return np.sqrt(total)


def calculate_rmse_intensity(exact: np.ndarray, approx: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between the intensity of the exit wave of the exact and approximate.
    """
    exact_intensity = calculate_farfield_intensity(exact)
    approx_intensity = calculate_farfield_intensity(approx)
    return calculate_rmse(exact_intensity, approx_intensity)


def calculate_farfield_intensity(exit_wave: np.ndarray):
    """
    Calculates the intensity of the exitwave by propagating the exitwave to the far field using a Fourier transform
    """
    # Propagate an exitwave to the far field using a Fourier transform
    # (FFT = Fast Fourier Transform Algorithm)
    farfield_wave = np.fft.fft(exit_wave)
    farfield_wave = np.fft.fftshift(farfield_wave)

    intensity = np.abs(farfield_wave) ** 2

    return intensity
