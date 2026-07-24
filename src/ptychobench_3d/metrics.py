# src/ptychobench/metrics.py
import numpy as np


def calculate_rmse(exact: np.ndarray, approx: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between two arrays.
    Uses pure NumPy vectorization for high speed and N-dimensional support.

    Formula: RMSE = sqrt( mean( |exact - approx|^2 ) )

    Parameters:
        exact (np.ndarray): The ground truth data array.
        approx (np.ndarray): The approximated data array.

    Returns:
        float: The calculated error.
    """
    return np.sqrt(np.mean(np.abs(exact - approx) ** 2))


def calculate_farfield_wave(
    exit_wave: np.ndarray, mode: str = "intensity"
) -> np.ndarray:
    """
    Calculates the farfield representation of the exitwave using an N-dimensional
    Fourier transform (fftn) to support both 1D and 2D transverse grids.
    """
    # Propagate an exitwave to the far field using a Fourier transform
    # fftn automatically handles 1D, 2D, or 3D arrays
    farfield_wave = np.fft.fftshift(np.fft.fftn(exit_wave))

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


def calculate_rmse_intensity(exact: np.ndarray, approx: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) between the intensity of the exit wave of the exact and approximate.
    """
    exact_intensity = calculate_farfield_wave(exact, mode="intensity")
    approx_intensity = calculate_farfield_wave(approx, mode="intensity")
    return calculate_rmse(exact_intensity, approx_intensity)


def calculate_max_intensity_error(exact: np.ndarray, approx: np.ndarray) -> float:
    """
    Calculates the maximum absolute error between the intensity of the exit wave of the exact and approximate.
    """
    exact_intensity = calculate_farfield_wave(exact, mode="intensity")
    approx_intensity = calculate_farfield_wave(approx, mode="intensity")
    return float(np.max(np.abs(exact_intensity - approx_intensity)))
