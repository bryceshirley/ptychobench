"""
Every figure this package draws.

Matplotlib lives here and nowhere else in the numerical path. Two reasons that
is worth a module of its own:

* **A figure is a host artefact.** Matplotlib speaks NumPy, so everything here
  converts with :func:`~ptychobench.utils.to_numpy` at the boundary. Keeping
  that in one module means the rest of the package never has to think about
  which backend a value came from.
* **Plotting is not what a result *is*.** :class:`~ptychobench.results.BenchmarkResult`
  is the run's record; drawing it is a separate concern with a separate
  dependency. The methods on ``BenchmarkResult`` and on ``Sample`` are thin
  delegators to the functions here, so existing calls keep working.

Functions take the object they draw as their first argument and return the
figure. None of them call ``plt.show()``: whether a figure is displayed, saved
or discarded is the caller's decision, and :mod:`ptychobench.reporting` depends
on being handed a figure rather than having one shown at it.
"""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from .numerics.metrics import calculate_farfield_wave
from .utils import to_numpy

if TYPE_CHECKING:  # pragma: no cover - import cycle broken for runtime
    from .results import BenchmarkResult
    from .samples import Sample

#: Line styles for the approximate operators, in order. Cycled rather than
#: indexed: indexing a fixed list by the loop counter raised ``IndexError``
#: past the seventh operator, and skipped a style whenever the reference was
#: not last.
STYLES: tuple[str, ...] = ("m-", "r--", "g-", "b-.", "c:", "y--", "C0-")

#: How the reference operator is drawn wherever it appears: thick, grey and
#: behind everything else, because it is the thing the others are measured
#: against rather than another competitor.
REFERENCE_STYLE = "k-"


def _styled(
    result: BenchmarkResult, operators: list[str] | None = None
) -> Iterator[tuple[str, np.ndarray, str, dict[str, Any]]]:
    """
    Yield ``(name, history, fmt, kwargs)`` for each operator to draw.

    Parameters
    ----------
    result : BenchmarkResult
        The run whose histories are drawn.
    operators : list of str, optional
        Class names to include, in order. All of them by default.

    Yields
    ------
    tuple
        ``(name, history, fmt, kwargs)``, with the reference styled thick,
        grey and behind everything else.

    Notes
    -----
    The one place the reference-highlighting rule is written. It used to be
    copy-pasted into four plotters, along with the style list, which is how
    they drifted apart.
    """
    styles = cycle(STYLES)
    names = list(result.wavefield_history) if operators is None else operators

    for name in names:
        label = {"label": result.label(name)}
        if name == result.REFERENCE:
            yield (
                name,
                result.wavefield_history[name],
                REFERENCE_STYLE,
                {
                    "linewidth": 4,
                    "alpha": 0.3,
                    **label,
                },
            )
        else:
            yield (
                name,
                result.wavefield_history[name],
                next(styles),
                {
                    "linewidth": 2,
                    **label,
                },
            )


def _history_extent(grid) -> list[float]:
    """
    Build the ``imshow`` bounding box for an ``(Nz, N)`` history array.

    Parameters
    ----------
    grid : SimulationGrid
        Supplies the transverse axis and the z steps.

    Returns
    -------
    list of float
        ``[left, right, bottom, top]``, to be paired with the default
        ``origin="upper"`` so that row ``0`` -- the field at ``z_steps[0]`` --
        lands at the top and z runs downward.

    Notes
    -----
    The three map plotters each wrote their own box and two of them were wrong
    in different ways. ``[-L/2, L/2, z_prop, 0]`` closes a domain that
    :attr:`~ptychobench.grid.SimulationGrid.x` leaves half-open and ends at a z
    the history never reaches. ``[x[0], x[-1], z_steps[0], z_steps[-1]]`` put
    ``bottom`` below ``top``, drawing the whole map upside down in z.
    """
    x = to_numpy(grid.x)
    z = grid.z_steps
    return [float(x[0]), float(x[-1]), float(z[-1]), float(z[0])]


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------


def plot_sample(result: BenchmarkResult) -> Figure:
    """
    Draw the sample's refractive index environment, modulus and phase.

    Parameters
    ----------
    result : BenchmarkResult
        The run whose recorded sample history is drawn.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    extent = _history_extent(result.grid)

    fig_env, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    im_real = ax_real.imshow(
        np.abs(result.sample_history), extent=extent, aspect="auto", cmap="viridis"
    )
    ax_real.set_title(r"Modulus of Sample", fontsize=14)
    ax_real.set_xlabel("Transverse coordinate x", fontsize=12)
    ax_real.set_ylabel("Propagation Distance z", fontsize=12)
    fig_env.colorbar(im_real, ax=ax_real)

    im_imag = ax_imag.imshow(
        np.angle(result.sample_history), extent=extent, aspect="auto", cmap="twilight"
    )
    ax_imag.set_title(r"Phase of Sample (radians)", fontsize=14)
    ax_imag.set_xlabel("Transverse coordinate x", fontsize=12)
    fig_env.colorbar(im_imag, ax=ax_imag)

    fig_env.suptitle("Refractive Index Environment", fontsize=16)
    return fig_env


def plot_wavefields(result: BenchmarkResult) -> tuple[Figure, Figure]:
    """
    Draw each operator's propagated field as a 2D map.

    Parameters
    ----------
    result : BenchmarkResult
        The run to draw.

    Returns
    -------
    phase : matplotlib.figure.Figure
        One panel per operator, showing the unwrapped phase.
    amplitude : matplotlib.figure.Figure
        One panel per operator, showing the modulus.
    """
    extent = _history_extent(result.grid)

    n_ops = len(result.wavefield_history)
    fig_p, axes_p = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)
    fig_a, axes_a = plt.subplots(1, n_ops, figsize=(6 * n_ops, 6), sharey=True)

    if n_ops == 1:
        axes_p, axes_a = [axes_p], [axes_a]

    for i, (name, data) in enumerate(result.wavefield_history.items()):
        im_p = axes_p[i].imshow(
            np.unwrap(np.angle(data), axis=0),
            extent=extent,
            aspect="auto",
            cmap="twilight",
        )
        axes_p[i].set_title(result.label(name), fontsize=14)
        axes_p[i].set_xlabel("Transverse coordinate x", fontsize=12)

        im_a = axes_a[i].imshow(
            np.abs(data), extent=extent, aspect="auto", cmap="magma"
        )
        axes_a[i].set_title(result.label(name), fontsize=14)
        axes_a[i].set_xlabel("Transverse coordinate x", fontsize=12)

    axes_p[0].set_ylabel("Propagation Distance z", fontsize=12)
    fig_p.colorbar(im_p, ax=axes_p, fraction=0.02, pad=0.02)
    fig_p.suptitle("2D Field Propagation (Phase)", fontsize=16)

    axes_a[0].set_ylabel("Propagation Distance z", fontsize=12)
    fig_a.colorbar(im_a, ax=axes_a, fraction=0.02, pad=0.02)
    fig_a.suptitle("2D Field Propagation (Amplitude)", fontsize=16)

    return fig_p, fig_a


def plot_evolution_1D(result: BenchmarkResult, x: float = 0.0) -> tuple[Figure, Figure]:
    """
    Draw phase and amplitude along z at one transverse position.

    Parameters
    ----------
    result : BenchmarkResult
        The run to draw.
    x : float, optional
        The transverse position, in the same units as ``grid.L``. Snapped to
        the nearest grid column by :meth:`~ptychobench.grid.SimulationGrid.index_of_x`.

    Returns
    -------
    phase : matplotlib.figure.Figure
        Unwrapped phase against z.
    amplitude : matplotlib.figure.Figure
        Modulus against z.

    Raises
    ------
    ValueError
        If ``x`` is outside the domain.
    """
    grid = result.grid
    # index_of_x rather than a formula written here. This plotter used to map x
    # with int((x + L/2) / L * (N - 1)) -- correct for a closed linspace, but
    # grid.x is half-open, so it disagreed with plot_evolution_slider about
    # which column "the slice at x" meant, for every x but the left edge.
    mid_index = grid.index_of_x(x)

    def comparison(extract, xlabel: str, ylabel: str, title: str) -> Figure:
        """
        Draw one quantity against z for every operator.

        Parameters
        ----------
        extract : callable
            Maps an ``(Nz, N)`` history to the ``(Nz,)`` curve to plot.
        xlabel, ylabel, title : str
            Axis labels and title.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        # On the Axes rather than the pyplot state machine, like the other seven
        # plotters. It worked either way -- each call opened a fresh figure --
        # but this was the one function whose output depended on which figure
        # happened to be current, so it was the one that would have broken if a
        # caller ever drew into a figure of their own first.
        fig, ax = plt.subplots(figsize=(10, 6))

        for _, data, fmt, kwargs in _styled(result):
            ax.plot(grid.z_steps, extract(data), fmt, **kwargs)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.tight_layout()
        return fig

    def phase(data: np.ndarray) -> np.ndarray:
        """
        Return the unwrapped phase along z at the chosen column.

        Parameters
        ----------
        data : numpy.ndarray
            An ``(Nz, N)`` history.

        Returns
        -------
        numpy.ndarray
            The ``(Nz,)`` phase curve.
        """
        return np.unwrap(np.angle(data[:, mid_index]), axis=0)

    def amplitude(data: np.ndarray) -> np.ndarray:
        """
        Return the modulus along z at the chosen column.

        Parameters
        ----------
        data : numpy.ndarray
            An ``(Nz, N)`` history.

        Returns
        -------
        numpy.ndarray
            The ``(Nz,)`` amplitude curve.
        """
        return np.abs(data[:, mid_index])

    fig1 = comparison(
        phase,
        "Propagation Distance z",
        "Phase (radians)",
        f"Phase Evolution along z at x = {x:.2f}",
    )
    fig2 = comparison(
        amplitude,
        "Propagation Distance z",
        r"Amplitude $|\psi|$",
        f"Amplitude Evolution along z at x = {x:.2f}",
    )
    return fig1, fig2


def plot_farfield_error(
    result: BenchmarkResult,
    log_scale: bool = False,
    mode: str = "intensity",
    operators: list[str] | None = None,
) -> Figure:
    """
    Draw each operator's far-field error against the reference.

    Parameters
    ----------
    result : BenchmarkResult
        The run to draw. Must include the reference operator.
    log_scale : bool, optional
        Logarithmic y-axis. Defaults to linear.
    mode : {"intensity", "magnitude", "phase"}, optional
        Which far-field quantity to difference.
    operators : list of str, optional
        Class names to include. All of them by default.

    Returns
    -------
    matplotlib.figure.Figure
        One error curve per approximate operator, against ``kx``.

    Raises
    ------
    ValueError
        If the reference operator was not part of the run, in which case there
        is nothing to measure an error against.
    """
    if result.REFERENCE not in result.wavefield_history:
        raise ValueError(
            "the reference operator is not in this run, so there is nothing "
            f"to plot an error against; got {sorted(result.wavefield_history)}"
        )

    grid = result.grid
    reference = to_numpy(
        calculate_farfield_wave(result.wavefield_history[result.REFERENCE][-1, :], mode)
    )
    # calculate_farfield_wave ends on an fftshift, so the abscissa is the
    # shifted transverse frequency -- not a real-space axis, which is what this
    # open-coded before.
    kx_values = to_numpy(grid.kx_shifted)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data, fmt, kwargs in _styled(result, operators):
        if name == result.REFERENCE:
            continue
        # to_numpy because this is a plotting path: Matplotlib needs a host
        # array, so a torch/MPS tensor has to come back across regardless.
        error = np.abs(reference - to_numpy(calculate_farfield_wave(data[-1, :], mode)))

        draw = ax.semilogy if log_scale else ax.plot
        draw(kx_values, error, fmt, label=kwargs["label"])

    ax.set_xlabel(r"Transverse Spatial Frequency $k_x$")
    ax.set_ylabel("Absolute Error")
    ax.set_title(
        f"Farfield {mode.capitalize()} vs Ground Truth {mode.capitalize()} Error"
    )
    ax.legend()
    return fig


def plot_evolution_slider(
    result: BenchmarkResult, x: float = 0.0
) -> tuple[Figure, dict[str, Slider]]:
    """
    The interactive counterpart of :func:`plot_evolution_1D`.

    Parameters
    ----------
    result : BenchmarkResult
        The run to draw.
    x : float, optional
        The initial transverse position.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The two-panel figure, with the slider axes below it.
    widgets : dict
        Holds the slider. Keep this alive in scripts and notebooks: Matplotlib
        holds only a weak reference to a Slider's callbacks.

    Raises
    ------
    ValueError
        If ``x`` is outside the domain.
    """
    grid = result.grid

    # One host copy of the transverse axis for the whole figure. The slider
    # callback would otherwise pull from the device on every drag event.
    x_axis = to_numpy(grid.x)

    def phase(data: np.ndarray, x_idx: int) -> np.ndarray:
        """
        Return the unwrapped phase along z at one column.

        Parameters
        ----------
        data : numpy.ndarray
            An ``(Nz, N)`` history.
        x_idx : int
            The column to slice.

        Returns
        -------
        numpy.ndarray
            The ``(Nz,)`` phase curve.
        """
        return np.unwrap(np.angle(data[:, x_idx]), axis=0)

    def amplitude(data: np.ndarray, x_idx: int) -> np.ndarray:
        """
        Return the modulus along z at one column.

        Parameters
        ----------
        data : numpy.ndarray
            An ``(Nz, N)`` history.
        x_idx : int
            The column to slice.

        Returns
        -------
        numpy.ndarray
            The ``(Nz,)`` amplitude curve.
        """
        return np.abs(data[:, x_idx])

    x_idx = grid.index_of_x(x)
    x_actual = x_axis[x_idx]

    fig, (ax_phase, ax_amp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(bottom=0.16, hspace=0.28)

    phase_lines: dict[str, Any] = {}
    amp_lines: dict[str, Any] = {}

    for name, data, fmt, kwargs in _styled(result):
        (phase_lines[name],) = ax_phase.plot(
            grid.z_steps, phase(data, x_idx), fmt, **kwargs
        )
        (amp_lines[name],) = ax_amp.plot(
            grid.z_steps, amplitude(data, x_idx), fmt, **kwargs
        )

    ax_phase.set_title(f"Phase Evolution along z at x = {x_actual:.4g}", fontsize=14)
    ax_phase.set_ylabel("Phase / radians", fontsize=12)
    ax_phase.grid(True, linestyle="--", alpha=0.7)
    ax_phase.legend(fontsize=10, loc="best")

    ax_amp.set_title(f"Amplitude Evolution along z at x = {x_actual:.4g}", fontsize=14)
    ax_amp.set_xlabel("Propagation Distance z", fontsize=12)
    ax_amp.set_ylabel(r"Amplitude $|\psi|$", fontsize=12)
    ax_amp.grid(True, linestyle="--", alpha=0.7)
    ax_amp.legend(fontsize=10, loc="best")

    slider_ax = fig.add_axes((0.15, 0.05, 0.70, 0.035))
    x_slider = Slider(
        ax=slider_ax,
        label="x",
        valmin=x_axis[0],
        valmax=x_axis[-1],
        valinit=x_actual,
        valstep=grid.dx,
    )

    def _update(x_value):  # pragma: no cover - needs a synthetic Matplotlib event
        """
        Redraw both panels at the slider's new transverse position.

        Parameters
        ----------
        x_value : float
            The position the slider moved to.
        """
        x_idx_new = grid.index_of_x(x_value)
        x_actual_new = x_axis[x_idx_new]

        for name, data in result.wavefield_history.items():
            phase_lines[name].set_ydata(phase(data, x_idx_new))
            amp_lines[name].set_ydata(amplitude(data, x_idx_new))

        ax_phase.set_title(
            f"Phase Evolution along z at x = {x_actual_new:.4g}", fontsize=14
        )
        ax_amp.set_title(
            f"Amplitude Evolution along z at x = {x_actual_new:.4g}", fontsize=14
        )

        for ax in (ax_phase, ax_amp):
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    x_slider.on_changed(_update)
    widgets = {"x_slider": x_slider}

    # Keep widgets alive in some notebook/script environments: Matplotlib holds
    # only a weak reference to a Slider's callbacks, so without an owning
    # reference the slider stops responding once `widgets` falls out of scope.
    # Set dynamically because Figure has no such declared field.
    setattr(fig, "_ptychobench_widgets", widgets)

    return fig, widgets


# ---------------------------------------------------------------------------
# Samples
#
# A sample is geometry, not a figure. These used to be methods on the Sample
# base class, which meant every subclass inherited a Matplotlib dependency to
# describe a permittivity profile.
# ---------------------------------------------------------------------------


def plot_sample_cross_section_modulus(
    sample: Sample, grid, z: float = 0.0, fourier: bool = False
) -> Figure:
    """
    Draw |eps(x)| at one z-slice, optionally in the Fourier domain.

    Parameters
    ----------
    sample : Sample
        The sample to evaluate.
    grid : SimulationGrid
        The grid to evaluate it on.
    z : float, optional
        The propagation distance of the slice. Defaults to ``0.0``.
    fourier : bool, optional
        Transform the slice before plotting. Defaults to False.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel figure.
    """
    eps = to_numpy(sample.get_permittivity(grid, z))
    if fourier:
        eps = np.fft.fftshift(np.fft.fft(eps))
        title = f"Modulus of Permittivity at z={z:.2f} (Fourier Domain)"
    else:
        title = f"Modulus of Permittivity at z={z:.2f} (Spatial Domain)"

    fig, ax = plt.subplots()
    ax.plot(to_numpy(grid.x), np.abs(eps))
    ax.set_xlabel("Transverse coordinate x")
    ax.set_ylabel("|ε(x, z)|")
    ax.set_title(title)
    return fig


def plot_sample_cross_section_phase(sample: Sample, grid, z: float = 0.0) -> Figure:
    """
    Draw the phase of eps(x) at one z-slice.

    Parameters
    ----------
    sample : Sample
        The sample to evaluate.
    grid : SimulationGrid
        The grid to evaluate it on.
    z : float, optional
        The propagation distance of the slice. Defaults to ``0.0``.

    Returns
    -------
    matplotlib.figure.Figure
        The single-panel figure.
    """
    eps = to_numpy(sample.get_permittivity(grid, z))

    fig, ax = plt.subplots()
    ax.plot(to_numpy(grid.x), np.angle(eps))
    ax.set_xlabel("Transverse coordinate x")
    ax.set_ylabel("∠ε(x, z)")
    ax.set_title(f"Phase of Permittivity at z={z:.2f}")
    return fig


def plot_sample_profile(sample: Sample, grid) -> Figure:
    """
    Draw the sample's full (x, z) permittivity map, modulus and phase.

    Parameters
    ----------
    sample : Sample
        The sample to evaluate. No propagation is needed, so a sample can be
        inspected before anything is run.
    grid : SimulationGrid
        The grid to evaluate it on.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    # A host record for Matplotlib, so each slice comes back off the device as
    # it is produced. Unlike `_TrajectoryRecorder` there is no propagation to
    # buffer against, so there is nothing to gain from blocking the copies.
    sample_history = np.zeros((grid.Nz, grid.N), dtype=np.complex128)
    for i, z in enumerate(grid.z_steps):
        sample_history[i, :] = to_numpy(sample.get_permittivity(grid, z))

    extent = _history_extent(grid)

    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    im_real = ax_real.imshow(
        np.abs(sample_history),
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    ax_real.set_title("Modulus of Sample")
    ax_real.set_xlabel("Transverse coordinate x")
    ax_real.set_ylabel("Propagation distance z")
    fig.colorbar(im_real, ax=ax_real)

    im_imag = ax_imag.imshow(
        np.unwrap(np.angle(sample_history), axis=1),
        extent=extent,
        aspect="auto",
        cmap="plasma",
    )
    ax_imag.set_title("Phase of Sample (radians)")
    ax_imag.set_xlabel("Transverse coordinate x")
    fig.colorbar(im_imag, ax=ax_imag)

    fig.suptitle("Refractive Index Environment")
    fig.tight_layout()
    return fig
