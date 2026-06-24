# Getting Started

Now that you have everything installed, it is time to run your first simulation! 

In `ptychobench`, running an experiment always follows the exact same 5-step recipe:
1. **Setup the Simulation Grid:** Define the physical space and the probe beam.
2. **Setup the Sample:** Choose the object you want to put under the microscope.
3. **Choose Operators:** Select which mathematical solvers will try to simulate the light.
4. **Run Benchmark:** compare the performance of different model operators.
5. **Generate Results:** Create the visual plots.

There are two example scripts included in the project that demonstrate this workflow.

---

## Example 1: Running a Single Experiment

The `example_single_sample.py` script runs a single simulation using the `Apoferritin` biological sample. 

Open `example_single_sample.py` to see how it is implemented in code. You can also modify the script to try different samples, operators, or grid parameters.

### How to run it:

Open your terminal, ensure you are in the project folder, and run the script using `uv`:

```bash
uv run example_single_sample.py
```

*Check your terminal for the printed error summary, and look in your results folder for the newly generated results!*

---

## Example 2: Running All Samples

The `example_all_samples.py` script runs a series of simulations for all the available samples. 

### How to run it:

Open your terminal, ensure you are in the project folder, and run the script using `uv`:

```bash
uv run example_all_samples.py
```
