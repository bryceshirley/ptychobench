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
    exact_intensity = calculate_farfield_wave(exact, mode="intensity")
    approx_intensity = calculate_farfield_wave(approx, mode="intensity")
    return calculate_rmse(exact_intensity, approx_intensity)


def calculate_farfield_wave(
    exit_wave: np.ndarray, mode: str = "intensity"
) -> np.ndarray:
    """
    Calculates the intensity of the exitwave by propagating the exitwave to the far field using a Fourier transform
    """
    # Propagate an exitwave to the far field using a Fourier transform
    # (FFT = Fast Fourier Transform Algorithm)
    farfield_wave = np.fft.fftshift(np.fft.fft(exit_wave))

    if mode == "intensity":
        return np.abs(farfield_wave) ** 2
    elif mode == "magnitude":
        return np.abs(farfield_wave)
    elif mode == "phase":
        return np.angle(farfield_wave)
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Choose from 'intensity', 'magnitude', or 'phase'."
        )


def calculate_max_intensity_error(exact: np.ndarray, approx: np.ndarray) -> float:
    """
    Calculates the maximum absolute error between the intensity of the exit wave of the exact and approximate.
    """
    exact_intensity = calculate_farfield_wave(exact, mode="intensity")
    approx_intensity = calculate_farfield_wave(approx, mode="intensity")
    return np.max(np.abs(exact_intensity - approx_intensity))
