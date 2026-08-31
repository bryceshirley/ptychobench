# Simulation Grid

Before you can run an experiment, you need to build the laboratory. In
`ptychobench`, the `SimulationGrid` is the digital sandbox where your physics
simulation takes place.

It defines the physical dimensions of the space, the resolution of the
simulation, the properties of the probe beam entering the sample, and which
array library and device the whole run executes on.

---

## The Simplest Possible Grid

Every parameter has a default, so this works:

```python
from ptychobench.grid import SimulationGrid

grid = SimulationGrid()
```

That gives you a 100-unit-wide domain at 200 pixels, propagating 100 units in
100 steps, with a collimated (non-diverging) beam of wavelength 0.1. In
practice you will want to set at least `N` and the physical scales to suit
your sample.

---

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `divergence_angle` | `float` | `0.0` | **Beam Spread:** How much the initial beam fans out, in degrees. `0.0` is a collimated beam. |
| `lam` | `float` | `0.1` | **Wavelength:** The wavelength of the light, in the same units as `L`. |
| `L` | `float` | `100.0` | **Domain Width:** The total transverse width of the simulation space. |
| `z_prop` | `float` | `100.0` | **Propagation Length:** How far the light travels from start to finish. |
| `N` | `int` | `200` | **Transverse Resolution:** How many pixels across the domain. |
| `Nz` | `int` | `100` | **Longitudinal Steps:** How many steps the solver takes to cross the sample. |
| `probe_width` | `float` | `5.0` | **Probe Size:** The Gaussian width of the beam before it spreads. |
| `backend` | `str` | `"numpy"` | **Array Library:** `"numpy"`, `"torch"` or `"cupy"`. See below. |
| `device` | `str \| None` | `None` | **Device:** `"cpu"`, `"mps"`, `"cuda"`, or `None` for the backend's own pick. |

**Example (customizing the grid):**

```python
# A high-resolution grid in a smaller physical space, with a diverging beam
grid = SimulationGrid(
    divergence_angle=10.0,
    L=50.0,
    N=256,
    Nz=200,
)
```

`SimulationGrid` is a dataclass, so you can also derive one grid from another
without repeating yourself:

```python
import dataclasses

fine = dataclasses.replace(grid, N=1024)
```

---

## Derived Quantities

You do not set these — the grid works them out from the parameters above.

| Property | Description |
| --- | --- |
| `k0` | Wavenumber, `2*pi / lam`. |
| `dx` | Transverse pixel size, `L / N`. |
| `dz` | Longitudinal step size, `z_prop / Nz`. |
| `x` | The transverse coordinates. |
| `kx` | The transverse Fourier coordinates, in FFT ordering. |
| `z_steps` | The propagation schedule: `Nz` positions from `0` to `z_prop`. |
| `propagating_mask` | `True` for Fourier modes with `abs(kx) <= k0`; `False` for evanescent ones. |

### `x` is half-open

The domain is `[-L/2, +L/2)` — the last column sits one pixel short of the
upper edge. That is what makes the discrete Fourier transform periodic, and it
is a real trap when you want "the slice at `x = 2.0`". Do not compute the index
yourself; ask the grid:

```python
column = grid.index_of_x(2.0)
```

`index_of_x` returns the nearest column, and raises `ValueError` for a position
outside the domain rather than silently clamping to the edge — a clamped answer
looks perfectly plausible and is for a position you never simulated.

---

## Choosing a Backend

The grid decides what the whole run executes on. Switch libraries with one
word:

```python
# NumPy, double precision. The accuracy reference.
grid = SimulationGrid(N=512, backend="numpy")

# Torch, on whatever accelerator is present
grid = SimulationGrid(N=512, backend="torch")

# Torch, pinned to the host -- which separates a *library* difference
# from a *device* one when you are comparing runs
grid = SimulationGrid(N=512, backend="torch", device="cpu")
```

`grid.resolved_device` tells you what `device=None` actually became, and
`grid.real_dtype` / `grid.complex_dtype` tell you the precision you got.

**Precision follows the device, not the library.** Torch on the CPU is
`float64`; torch on Apple's MPS is `float32`, because Metal has no
double-precision support at all. That is not a bug you can configure around,
and it puts a noise floor of roughly `1e-4` under every error the benchmark
reports — enough to swamp the difference between the two third-order operators.
Use NumPy, or torch pinned to the CPU, when you care about the accuracy
numbers; use the GPU when you care about throughput at large `N`.

### What the backend does and does not govern

It governs the *field* arrays: `x`, `kx`, `propagating_mask`, the two Fourier
symbols and `get_initial_field()` — everything the matrix-free integrators
touch.

Two things deliberately stay on the host whatever you set:

* `z_steps`, which is a schedule driving a Python loop rather than a field.
* The dense `N x N` matrices (`F`, `F_inv`, `get_kinetic_operator()`,
  `get_angular_spectrum_operator()`), because the `"direct"` integrator hands
  them to SciPy's `expm` and `sqrtm`, which are host-only.

That second point does **not** mean `"direct"` fails on a GPU grid — the field
is carried to the host and back for you, and `step` still returns an array in
the grid's backend. It does mean `"direct"` is the slow path there, and pays a
transfer per step on top of its `O(N^3)`.

A benchmark's recorded histories always come back as host NumPy `complex128`,
so your metrics and plots never see a device array regardless of what the
propagation ran on. See `examples/full_backend_demo.py` for a complete run on
torch.
