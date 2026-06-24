# Benchmark Results Guide

After `run_benchmark()` completes, it returns a `BenchmarkResult` object that contains all the data you need to analyse your simulation. This object allows access to the RMSE error metrics, the propagated wavefields, and the sample structure. It also has built-in methods to generate plots and summaries of your results.

---

## The `BenchmarkResult` Object

The result object contains everything you need to reproduce the experiment and analyze the data. 

| Property | Type | Description |
| :--- | :--- | :--- |
| **`errors`** | `dict` | A dictionary mapping the **Operator Class Name** (e.g., `'ParaxialOperator'`) to its calculated RMSE. |
| **`sample_name`** | `str` | The name of the sample you tested (e.g., `'Apoferritin'`). |
| **`sample_params`** | `dict` | A dictionary of the parameters you passed to the sample (e.g., `{'modulus': 0.8}`). |
| **`grid`** | `SimulationGrid` | The physical grid configuration used for the run. |

### The Propagation Data

* **`wavefield_history`**: A dictionary containing the full 2D propagated wavefield through the sample for each operator.
* **`sample_history`**: The 2D physical structure of the sample the light wavefield passed through.

---

## 1. Quick Console Summary

If you just want a quick look at how your operators performed, the results object comes with a built-in helper method to print a clean summary to your terminal.

```python
from ptychobench.grid import SimulationGrid
from ptychobench.samples import ZigBalls
from ptychobench.operators import ParaxialOperator, FeitFleckOperator
from ptychobench.solver import run_benchmark

# 1. Setup your simulation
grid = SimulationGrid(divergence_angle=15.0)
sample = ZigBalls(modulus=0.5)

# 2. Run Benchmark for the two operators you want to compare
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

Errors (RMSE relative to ExactOperator):
  FeitFleckOperator   : 4.567000e-05
  ParaxialOperator    : 1.234500e-04
---------------------------------------------

```

---

## 2. Generating Plots and Saving Results

The result object comes with `generate_plots()` method that builds a experimental report.

```python
# Create a timestamped folder with all graphs and logs
result.generate_plots()

# Or specify a custom directory:
result.generate_plots(save_dir="my_results_folder")
```

When you call this method, it creates a new time-stamped folder (e.g., `results/run_ZigBalls_2026-06-24_14-30-00/`) to ensure you never accidentally overwrite old data.

**Inside this folder, it automatically generates:**

* `simulation_summary.txt`: A clean text file recording all your grid and sample parameters.
* `quantitative_errors.txt`: A text file containing the exact RMSE values.
* `[SampleName].png`: A 2D plot of your sample's Modulus and Phase environment.
* `propagated_fields_2D_phase.png`: A side-by-side comparison of the 2D phase wavefields for all operators.
* `propagated_fields_2D_amp.png`: A side-by-side comparison of the 2D amplitude wavefields.
* `phase_evolution_1D.png` & `amplitude_evolution_1D.png`: 1D line graphs showing how the wavefield evolves precisely down the center of the grid.

---

## 3. Accessing Errors

You will eventually want to write your own scripts to analyse the error data (for example, to build convergence plots across multiple runs).

You can extract the raw error numbers safely using the `get_error()` and `get_errors()` methods.

### Example: Extracting Error Data for a Single Operator

```python
# --- Getting a single error ---
paraxial_error = result.get_error("ParaxialOperator")

# Print the error
print(f"Paraxial Operator Error: {paraxial_error:.6e}")

```

### Example: Extracting All Errors

```python
# --- Getting all errors as a dictionary ---
all_errors = result.get_errors()

# Print the errors in a formatted way, skipping the exact baseline
print("\nExtracting Data for Custom Analysis:")
for operator_name, error_value in all_errors.items():
    
    if operator_name == "ExactOperator":
        continue
        
    print(f"{operator_name}: {error_value:.6e}")

```
