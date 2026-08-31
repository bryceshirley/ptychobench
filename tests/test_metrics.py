"""
=================================================================
PING-PONG Test Driven Development: WRITING YOUR OWN METRICS TESTS
=================================================================
In this exercise, you will practice Test-Driven Development (TDD) using the
"Ping-Pong" pairing method.

WHAT TO DO (THE RULES):
1. Student A: Writes a failing test (e.g., test_rmse_identical_arrays).
2. Student B: Writes just enough code in `metrics.py` to make it pass.
3. Student B: Writes the next failing test (e.g., test_rmse_known_real_difference).
4. Student A: Writes code in `metrics.py` to make it pass.

HOW TO SET UP ARRAYS:
Use the `numpy` library to create your lists of data.
Example: `my_array = np.array([1.0, 2.0, 3.0])`

HOW TO WRITE A PYTEST:
A test is just a standard Python function that starts with the word `test_`.
It follows the "Arrange-Act-Assert" pattern:
- Arrange: Set up your variables (e.g., create `exact` and `approx` arrays).
- Act: Call the function you are testing (e.g., `error = calculate_rmse(...)`).
- Assert: Check if the result matches your math using the `assert` keyword.
  Example: `assert error == 0.0, "Expected RMSE to be 0!"`

COMPLEX NUMBERS IN PYTHON:
Usually in maths, we use 'i' for imaginary numbers. Python uses 'j'.
Example of a complex number: `val = 3.0 + 2.0j`

EDGE CASES TO THINK ABOUT:
What happens if the arrays are different lengths? What if they are empty?
Good code doesn't just calculate the right answer; it handles bad inputs safely
by raising clear errors.
"""

import pytest

from ptychobench.numerics.metrics import (
    calculate_rmse,
    calculate_farfield_wave,
    calculate_max_error,
    calculate_max_intensity_error,
    calculate_rmse_intensity,
)
import numpy as np


def test_rmse_identical_arrays():
    """
    TDD STEP 1: The Sanity Check.
    If the exact array and the approximation array are perfectly identical,
    the error must be exactly 0.0.
    """
    exact = np.array([1, 2, 3])
    approx = np.array([1, 2, 3])
    assert 0.0 == calculate_rmse(exact, approx)


def test_rmse_known_real_difference():
    """
    TDD STEP 2: The Hand-Calculated Test.
    Use simple real numbers that can be calculated on a piece of paper.
    """
    list_x = np.array([4.0, 4.0, 2.0])
    list_y = np.array([2.0, 6.0, 0.0])
    # assert len(list_x) == len(list_y)
    print(calculate_rmse(list_x, list_y))
    assert 2.0 == calculate_rmse(list_x, list_y)


# ------------------------------------------------------------------------------
# EDGE CASES
# ------------------------------------------------------------------------------
# Are there any edge cases you can think of?
# Error assertions look like this:
# with pytest.raises(ValueError):
#      calculate_rmse(exact, approx)


def test_rmse_of_complex_arrays_uses_the_magnitude_of_the_difference():
    """Wavefields are complex, so the error is |exact - approx|, not the
    difference of the real parts. A 3-4-5 triangle makes that visible: a
    difference of 3 + 4j has magnitude 5, which the real part alone would miss.
    """
    exact = np.array([3.0 + 4.0j])
    approx = np.array([0.0 + 0.0j])
    assert 5.0 == calculate_rmse(exact, approx)


#: Each far-field wrapper and the plain metric it is the far-field form of.
#: Kept as a pair because that is the invariant: a wrapper transforms both
#: arguments and delegates, so it must agree with transforming first and calling
#: the plain form -- which is the cheaper route `run_benchmark` takes, and the
#: only thing keeping the two paths from drifting apart.
INTENSITY_PAIRS = [
    (calculate_rmse_intensity, calculate_rmse),
    (calculate_max_intensity_error, calculate_max_error),
]


@pytest.mark.parametrize("wrapper, plain", INTENSITY_PAIRS, ids=lambda f: f.__name__)
def test_the_far_field_metrics_are_zero_for_identical_arrays(wrapper, plain):
    values = np.array([8.0, 16.0, 22.0])
    assert 0.0 == wrapper(values, values)


@pytest.mark.parametrize("wrapper, plain", INTENSITY_PAIRS, ids=lambda f: f.__name__)
def test_the_far_field_wrapper_agrees_with_transforming_first(wrapper, plain):
    exact = np.array([8.0, 16.0, 22.0])
    approx = np.array([8.0, 12.0, 120.0])

    expected = plain(
        calculate_farfield_wave(exact, mode="intensity"),
        calculate_farfield_wave(approx, mode="intensity"),
    )
    assert wrapper(exact, approx) == expected
    assert expected > 0.0


def test_zero_intensity():
    list = np.array([0.0, 0.0, 0.0])
    assert 0.0 == np.sum(calculate_farfield_wave(list))


def test_non_zero_intensity():
    list_x = np.array([8.0, 16.0, 22.0])
    assert np.sum(calculate_farfield_wave(list_x)) > 0.0


# ------------------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------------------
# test_rmse_identical_arrays above passes integer arrays, which NumPy quietly
# promotes on `mean`. The Array API leaves `mean` undefined for integers and
# torch raises outright, so the same two lines are worth running on a second
# backend -- otherwise the promotion NumPy does for free reads as behaviour the
# package provides.


def test_rmse_of_integer_arrays_works_on_any_backend():
    """An RMSE is a real number whatever was differenced."""
    torch = pytest.importorskip("torch", reason="the torch backend is optional")

    exact = torch.asarray([4, 4, 2])
    approx = torch.asarray([2, 6, 0])

    assert calculate_rmse(exact, approx) == 2.0
