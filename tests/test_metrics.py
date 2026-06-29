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

from ptychobench.metrics import calculate_rmse
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
