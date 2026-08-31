"""
Error metrics, and the far-field transform they are usually measured after.

Arrays in, floats out. Nothing here knows about a grid, which is why it sits
beside the FFT pair rather than beside the benchmark that calls it.

Two forms of each metric. :func:`calculate_rmse` and :func:`calculate_max_error`
compare the arrays they are handed; the ``*_intensity`` forms transform both to
the far field first and then delegate. The plain forms are the ones to reach for
if you want more than one number off the same pair, because each wrapper
transforms independently and the transforms are not shared between them.
"""

from typing import Any

from array_api_compat import array_namespace

from .transforms import fft_namespace
from ..utils import Array


def calculate_rmse(exact: Array, approx: Array) -> float:
    """
    Root mean square error between two arrays.

    ``RMSE = sqrt( (1/N) * sum( |exact_i - approx_i|^2 ) )``

    Parameters
    ----------
    exact : Array
        The ground truth data array.
    approx : Array
        The approximated data array. Must be the same shape as ``exact``.

    Returns
    -------
    float
        The error.

    Raises
    ------
    ValueError
        If the two arrays have different shapes.
    """
    xp = array_namespace(exact)  # Ensure the namespace is consistent

    # Ensure both arrays have the same shape
    if exact.shape != approx.shape:
        raise ValueError(
            f"Input arrays must have the same shape. Got {exact.shape} and {approx.shape}."
        )
    # Calculate the squared differences
    squared_diff = xp.abs(exact - approx) ** 2

    # `mean` is undefined for integer arrays in the Array API, and torch raises
    # rather than promoting ("could not infer output dtype"). An RMSE is a real
    # number whatever was differenced, so cast first. The target comes from the
    # namespace's own default for that device, which is the only way to ask for
    # "the sensible float here" without assuming float64 exists -- Metal has no
    # FP64. Integer inputs are a robustness path, not the measurement path:
    # every field this package propagates is already complex.
    if not xp.isdtype(squared_diff.dtype, "real floating"):
        defaults = xp.__array_namespace_info__().default_dtypes(
            device=squared_diff.device
        )
        squared_diff = xp.astype(squared_diff, defaults["real floating"])

    # Calculate the mean of the squared differences
    mean_squared_diff = xp.mean(squared_diff)
    # float() rather than returning the 0-d array: the annotation promises a
    # scalar, and callers put this straight into f-strings and sorts.
    return float(xp.sqrt(mean_squared_diff))


def calculate_rmse_intensity(exact: Array, approx: Array) -> float:
    """
    Root mean square error between two exit waves' far-field intensities.

    A convenience wrapper: it transforms both arguments and delegates to
    :func:`calculate_rmse`. A caller that wants more than one metric off the
    same pair should transform once with :func:`calculate_farfield_wave`
    instead, which is what ``run_benchmark`` does.

    Parameters
    ----------
    exact : Array
        The ground truth exit wave.
    approx : Array
        The approximated exit wave.

    Returns
    -------
    float
        The error, in intensity units.
    """
    exact_intensity = calculate_farfield_wave(exact, mode="intensity")
    approx_intensity = calculate_farfield_wave(approx, mode="intensity")
    return calculate_rmse(exact_intensity, approx_intensity)


def calculate_farfield_wave(exit_wave: Array, mode: str = "intensity") -> Any:
    """
    Propagate an exit wave to the far field and take one of its parts.

    The far field is one Fourier transform away, so this is the detector plane:
    an unnormalised, shifted FFT. Unnormalised on purpose -- unlike the unitary
    pair in :mod:`~ptychobench.numerics.transforms` -- because these feed error
    metrics, and rescaling them would move every number the benchmark reports.

    Parameters
    ----------
    exit_wave : Array
        The field at the end of the propagation.
    mode : str, optional
        ``"intensity"``, ``"magnitude"`` or ``"phase"``.

    Returns
    -------
    Array
        The requested part of the far field, real-valued in every mode.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three.
    """
    # Routed through fft_namespace rather than xp.fft directly: fft is an
    # optional extension in the Array API, so a conforming backend need not
    # have it, and the shared guard says so clearly instead of raising
    # AttributeError.
    xp = array_namespace(exit_wave)
    fft_ext = fft_namespace(exit_wave)
    farfield_wave = fft_ext.fftshift(fft_ext.fft(exit_wave))

    if mode == "intensity":
        return xp.abs(farfield_wave) ** 2
    elif mode == "magnitude":
        return xp.abs(farfield_wave)
    elif mode == "phase":
        return xp.angle(farfield_wave)
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Choose from 'intensity', 'magnitude', or 'phase'."
        )


def calculate_max_error(exact: Array, approx: Array) -> float:
    """
    Largest absolute difference between two arrays.

    Parameters
    ----------
    exact : Array
        The ground truth data array.
    approx : Array
        The approximated data array.

    Returns
    -------
    float
        The error.
    """
    xp = array_namespace(exact)
    return float(xp.max(xp.abs(exact - approx)))


def calculate_max_intensity_error(exact: Array, approx: Array) -> float:
    """
    Largest absolute error between two exit waves' far-field intensities.

    The counterpart of :func:`calculate_rmse_intensity`, with the same caveat
    about transforming twice.

    Parameters
    ----------
    exact : Array
        The ground truth exit wave.
    approx : Array
        The approximated exit wave.

    Returns
    -------
    float
        The error, in intensity units.
    """
    exact_intensity = calculate_farfield_wave(exact, mode="intensity")
    approx_intensity = calculate_farfield_wave(approx, mode="intensity")
    return calculate_max_error(exact_intensity, approx_intensity)
