import numpy as np


# ==========================================
# Private Helper Functions (Pure Math)
# ==========================================
def _apply_boundary_mask(eps: np.ndarray, x_array: np.ndarray, L: float) -> np.ndarray:
    """Forces free space (epsilon perturbation = 0) in the padded boundary regions."""
    eps[np.abs(x_array) >= L / 2] = 0.0 + 0j
    return eps


def _build_waveguides(x_array, modulus, phase, offset, period, core_width):
    """Core logic for generating periodic waveguides using modulus and phase."""
    x_mod = (x_array - offset + period / 2) % period - period / 2
    profile = np.exp(-(x_mod**2) / (core_width**2))

    # Convert polar (modulus, phase) to a complex perturbation
    complex_pert = modulus * np.exp(1j * phase)
    return complex_pert * profile


def _build_balls(
    x_array,
    z,
    modulus,
    phase,
    z_prop,
    num_balls,
    ball_radius,
    x_period,
    x_amplitude,
    is_zigzag,
):
    """Core logic for generating localized spherical features using array broadcasting."""
    z_centers = np.linspace(0, z_prop, num_balls)

    if is_zigzag:
        cycle_pos = (np.arange(num_balls) % 8) / 8.0
        x_shift = np.piecewise(
            cycle_pos,
            [
                cycle_pos <= 0.25,
                (cycle_pos > 0.25) & (cycle_pos <= 0.75),
                cycle_pos > 0.75,
            ],
            [
                lambda c: 4.0 * c,
                lambda c: 1.0 - 4.0 * (c - 0.25),
                lambda c: -1.0 + 4.0 * (c - 0.75),
            ],
        )
        x_offsets = x_amplitude * x_shift
    else:
        x_offsets = np.zeros(num_balls)

    x_mod = (
        x_array[None, :] - x_offsets[:, None] + x_period / 2
    ) % x_period - x_period / 2
    z_diff = z - z_centers[:, None]

    profiles = np.exp(-(x_mod**2 + z_diff**2) / (ball_radius**2))
    total_profile = np.sum(profiles, axis=0)

    # Convert polar (modulus, phase) to a complex perturbation
    complex_pert = modulus * np.exp(1j * phase)
    return complex_pert * total_profile


# ==========================================
# Public API (What the students call)
# ==========================================


def straight(
    grid, z: float, modulus=0.08, phase=None, period=10.0, core_width=1.5, **kwargs
):
    """A periodic array of perfectly straight waveguides."""
    if phase is None:
        phase = modulus * 1e-2
    eps = _build_waveguides(
        grid.x, modulus, phase, offset=0.0, period=period, core_width=core_width
    )
    return _apply_boundary_mask(eps, grid.x, grid.L)


def zigs(
    grid,
    z: float,
    modulus=0.08,
    phase=None,
    period=10.0,
    core_width=1.5,
    wiggle_amp=2.0,
    **kwargs,
):
    """A periodic array of waveguides that curve (wiggle) along the z-axis."""
    if phase is None:
        phase = modulus * 1e-2
    wiggle_offset = wiggle_amp * np.sin(2 * np.pi * z / (grid.z_prop / 2))
    eps = _build_waveguides(
        grid.x,
        modulus,
        phase,
        offset=wiggle_offset,
        period=period,
        core_width=core_width,
    )
    return _apply_boundary_mask(eps, grid.x, grid.L)


def straight_balls(
    grid,
    z: float,
    modulus=0.08,
    phase=None,
    num_balls=15,
    ball_radius=0.8,
    x_period=5.0,
    **kwargs,
):
    """A discrete sequence of highly localized spherical features arranged in perfectly straight vertical lines."""
    if phase is None:
        phase = modulus * 1e-2
    eps = _build_balls(
        grid.x,
        z,
        modulus,
        phase,
        grid.z_prop,
        num_balls,
        ball_radius,
        x_period,
        x_amplitude=0.0,
        is_zigzag=False,
    )
    return _apply_boundary_mask(eps, grid.x, grid.L)


def zig_balls(
    grid,
    z: float,
    modulus=0.08,
    phase=None,
    num_balls=25,
    ball_radius=0.8,
    x_period=5.0,
    x_amplitude=3.0,
    **kwargs,
):
    """A discrete sequence of highly localized spherical features arranged in a sharp zig-zag pattern."""
    if phase is None:
        phase = modulus * 1e-2
    eps = _build_balls(
        grid.x,
        z,
        modulus,
        phase,
        grid.z_prop,
        num_balls,
        ball_radius,
        x_period,
        x_amplitude,
        is_zigzag=True,
    )
    return _apply_boundary_mask(eps, grid.x, grid.L)
