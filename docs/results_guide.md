# Benchmark Results Guide

`run_benchmark()` returns a `BenchmarkResult`: the record of what the run
produced. It holds the error metrics, the propagated wavefields and the sample
structure, and it can draw and save all of them for you.

One thing to know before anything else. **Every per-operator dictionary on a
result is keyed by the operator's class name** — `"ParaxialOperator"`, not
`"Paraxial"`. The pretty labels you see in plot legends live in
`display_names` and nowhere else, so `result.rmse_wavefield["Paraxial"]` is a
`KeyError`, not a number.

The histories come back as host NumPy `complex128` whatever backend the run
executed on, so nothing downstream has to care whether the propagation ran on a
GPU.

---

## The `BenchmarkResult` Object

| Property | Type | Description |
| :--- | :--- | :--- |
| **`rmse_wavefield`** | `dict[str, float]` | Class name → RMSE of the exit wavefield against the reference operator. |
| **`rmse_detector`** | `dict[str, float]` | Class name → RMSE of the far-field intensity against the reference. |
| **`max_error_detector`** | `dict[str, float]` | Class name → largest absolute far-field intensity error against the reference. |
| **`sample_name`** | `str` | The sample class name, e.g. `'Apoferritin'`. |
| **`sample_params`** | `dict` | The parameters the sample was built with, e.g. `{'modulus': 0.8}`. |
| **`grid`** | `SimulationGrid` | The grid the run used. |
| **`wavefield_history`** | `dict[str, ndarray]` | Class name → the full `(Nz, N)` complex field through the sample. |
| **`sample_history`** | `ndarray` | The `(Nz, N)` complex permittivity the light passed through. |
| **`display_names`** | `dict[str, str]` | Class name → plot label, e.g. `{'FeitFleckOperator': 'Feit/Fleck'}`. |
| **`REFERENCE`** | `str` | The class name errors are measured against: `"GroundTruthOperator"`. |

`result.label("FeitFleckOperator")` gives you the display label, falling back to
the class name if the run did not record one.

---

## 1. Quick Console Summary

```python
from ptychobench.grid import SimulationGrid
from ptychobench.samples import ZigBalls
from ptychobench.operators import ParaxialOperator, FeitFleckOperator
from ptychobench.benchmark import run_benchmark

# 1. Setup your simulation
grid = SimulationGrid(N=512, L=40.0, z_prop=20.0, Nz=100)
sample = ZigBalls(modulus=0.5)

# 2. Run the benchmark for the two operators you want to compare.
#    GroundTruthOperator is added for you: it is the reference, so a run
#    without it has nothing to measure against.
result = run_benchmark(grid, sample, [ParaxialOperator, FeitFleckOperator])

# 3. Print a formatted summary to your console
result.print_summary()
```

**Example Output:**

```text
--- Benchmark Summary: ZigBalls ---
Parameters:
  modulus     : 0.5
  num_balls   : 25
  x_amplitude : 3.0

 Wavefield Errors (RMSE relative to GroundTruthOperator):
  Feit/Fleck          : 4.567000e-05
  Paraxial            : 1.234500e-04
---------------------------------------------

Detector Intensity Errors (RMSE relative to GroundTruthOperator):
  Feit/Fleck          : 8.910000e-04
  Paraxial            : 2.345600e-03
---------------------------------------------

Detector Intensity Errors (Max Absolute Error relative to GroundTruthOperator):
  Feit/Fleck          : 1.020000e-02
  Paraxial            : 3.040000e-02
---------------------------------------------
```

Three error tables, not one. The wavefield RMSE is the honest measure of how
well an operator propagated the field; the two detector rows are what a
ptychography experiment would actually see, since a detector measures intensity
and throws the phase away.

---

## 2. Saving a Report

`generate_report()` writes everything to a fresh timestamped directory, so you
cannot accidentally overwrite an earlier run. It returns the `Path` it wrote to.

```python
# Create a timestamped folder with every figure and both text summaries
run_dir = result.generate_report()

# Or specify a custom parent directory:
run_dir = result.generate_report(save_dir="my_results_folder")

print(f"written to {run_dir}")
```

The directory is named `results/run_<SampleName>_<YYYY-MM-DD_HH-MM-SS>/`, and
inside it you get exactly these files:

* `simulation_summary.txt` — every grid and sample parameter, plus the backend
  and device the run executed on.
* `errors.md` — a Markdown table of all three error metrics, one row per
  operator.
* `<SampleName>.png` — the sample's modulus and phase over the full `(x, z)`
  map.
* `propagated_fields_2D_amp.png` and `propagated_fields_2D_phase.png` — the 2D
  fields, side by side across operators.
* `phase_evolution_1D.png` and `amplitude_evolution_1D.png` — line plots down
  the centre of the grid.
* `farfield_error_intensity.png`, `farfield_error_magnitude.png` and
  `farfield_error_phase.png` — the far-field error for each operator, in all
  three modes.

---

## 3. Plotting

Every plot method **returns** its figures rather than only showing them, so you
can adjust them, save them yourself, or embed them. In a Jupyter notebook the
figure displays as usual; in a script, call `matplotlib.pyplot.show()`.

#### The 2D wavefield evolution

```python
# Returns two figures, phase first
fig_phase, fig_amp = result.plot_wavefields()
```

#### The 1D wavefield evolution

```python
# Phase and amplitude down the centre of the grid (x = 0.0)
fig_phase, fig_amp = result.plot_evolution_1D()

# Or at any x inside the domain [-L/2, L/2).
# x is snapped to the nearest grid column; outside the domain raises ValueError.
fig_phase, fig_amp = result.plot_evolution_1D(x=2.0)
```

#### Slider evolution plot — for Jupyter notebooks

```python
%matplotlib widget
fig, widgets = result.plot_evolution_slider()

# `widgets` keeps the Slider alive; drop it and the callback is garbage
# collected and the slider stops responding.
```

#### The sample structure

```python
fig = result.plot_sample()
```

#### The error across the far field

Both need `GroundTruthOperator` in the run, which `run_benchmark` guarantees.

```python
# The default: intensity error, on a linear scale, for every operator
fig = result.plot_farfield_error()

# A log scale is usually what you want -- the operators here differ by
# orders of magnitude, not by percentages.
fig = result.plot_farfield_error(log_scale=True)

# `mode` picks what is compared: "intensity" (default), "magnitude" or "phase"
fig = result.plot_farfield_error(mode="magnitude")

# And you can restrict the comparison to a subset, by class name
fig = result.plot_farfield_error(operators=["ParaxialOperator"])
```

---

## 4. Accessing Errors

To write your own analysis — a convergence plot across several runs, say — pull
the raw numbers out.

```python
# One operator, by class name. Raises KeyError, listing the operators that
# *were* in the run, if the name is not one of them. It used to return inf,
# which sorts and formats like a real error and so turned a typo into a
# plausible number.
paraxial_error = result.get_rmse_wavefield("ParaxialOperator")
print(f"Paraxial Operator Error: {paraxial_error:.6e}")

# The whole dictionary
all_errors = result.get_rmse_wavefield()

# The detector-side equivalents
detector_error = result.get_rmse_detector("ParaxialOperator")
all_detector_errors = result.get_rmse_detector()
```

```python
print("\nExtracting Data for Custom Analysis:")
for operator_name, error_value in result.get_rmse_wavefield().items():
    if operator_name == result.REFERENCE:
        continue  # the reference has no error against itself
    print(f"{result.label(operator_name)}: {error_value:.6e}")
```

`get_rmse_wavefield_dict()` and `get_rmse_detector_dict()` are also available
and return the same dictionaries as the no-argument calls above.

---

## 5. Where the Code Lives

`BenchmarkResult` is the run's record. Drawing and saving are separate modules,
and the methods above are one-line forwards into them — so if you want to plot
something the result object does not offer, you can call the underlying
functions directly.

| Module | Holds |
| :--- | :--- |
| `ptychobench.results` | `BenchmarkResult`: the data, the accessors, the summary. |
| `ptychobench.plotting` | Every figure the package draws. Pure functions taking a result (or a sample and a grid), returning figures. Nothing here calls `show()`. |
| `ptychobench.reporting` | `BenchmarkReport`: turns a result into a directory of files. |

```python
from ptychobench import plotting

# The same figure result.plot_farfield_error() returns
fig = plotting.plot_farfield_error(result, log_scale=True)

# And plotters that have no method on BenchmarkResult, because they need a
# sample rather than a run:
fig = plotting.plot_sample_profile(sample, grid)
fig = plotting.plot_sample_cross_section_modulus(sample, grid, z=0.0)
fig = plotting.plot_sample_cross_section_modulus(sample, grid, z=0.0, fourier=True)
fig = plotting.plot_sample_cross_section_phase(sample, grid, z=0.0)
```

Matplotlib is imported in `plotting` and nowhere else in the package, and every
plotter converts its arrays to NumPy at the boundary — which is why a run on a
GPU backend plots exactly like a run on NumPy.
