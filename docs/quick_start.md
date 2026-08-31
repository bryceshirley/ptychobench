# Getting Started

Now that you have everything installed, it is time to run your first
simulation.

In `ptychobench`, running an experiment always follows the same five-step
recipe:

1. **Set up the simulation grid** — the physical space and the probe beam.
2. **Set up the sample** — the object you are putting under the microscope.
3. **Choose operators** — which approximations you want to compare.
4. **Run the benchmark** — propagate the field with each of them.
5. **Look at the results** — plots, a printed summary, a saved report.

---

## The Tutorial Notebooks

The main worked example is the Jupyter notebook at the top of the repository, 
`benchmark_tutorial.ipynb`.
---

## The Five Steps, in a Script

```python
from ptychobench.grid import SimulationGrid
from ptychobench.samples import Apoferritin
from ptychobench.operators import (
    ParaxialOperator,
    FeitFleckOperator,
    YevickThomsonOperator,
    LinDudaOperator,
)
from ptychobench.benchmark import run_benchmark

# 1. The sandbox
grid = SimulationGrid(
    divergence_angle=4.0,  # degrees of beam spread
    lam=0.00197,           # 300 keV electrons, in nm
    L=12.8,                # nm across
    z_prop=20.0,           # nm of propagation
    N=512,                 # transverse pixels
    Nz=100,                # propagation steps
    probe_width=1.0,       # nm
)

# 2. The object
sample = Apoferritin(modulus=0.08)

# 3. The approximations to compare. GroundTruthOperator is added for you --
#    it is the reference every error is measured against.
operators = [
    ParaxialOperator,
    FeitFleckOperator,
    YevickThomsonOperator,
    LinDudaOperator,
]

# 4. Propagate
result = run_benchmark(grid, sample, operators, integrator="krylov")

# 5. Look at what happened
result.print_summary()
result.plot_wavefields()
result.plot_farfield_error(log_scale=True)

# ...and save it all to a timestamped directory
run_dir = result.generate_report()
print(f"written to {run_dir}")
```

In a script, add `matplotlib.pyplot.show()` at the end to keep the figure
windows open — every plot method returns its figures rather than blocking.

---

## Turning On the Logs

`ptychobench` is a library, so it logs but leaves the formatting and the
destination to you. One line switches its progress messages on:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
```

Worth doing: it is how the benchmark tells you when it has fallen back from
split-step to Krylov for the reference operator, which has no split form.

---

## Choosing an Integrator

The `integrator` argument decides *how* the exponential is evaluated, which is
a separate question from which approximation you are testing.

* `"direct"` (the default) forms the dense matrix and is the accuracy
  reference. `O(N^3)` — fine up to a few hundred pixels, painful beyond.
* `"krylov"` is matrix-free and works for every operator. This is the one to
  reach for at realistic `N`.
* `"split-step"` is the fastest, and every one of the four approximations
  supports it — H1 and H2 because they split into the two diagonal pieces it
  needs, H3 and H4 through an extra factor for their cross terms. It is also
  the only integrator that carries gradients. The ground-truth reference has no split
  form, so the benchmark falls back to Krylov for it and logs that it has.

---

## Running Somewhere Else

The grid decides which array library and device the whole run executes on:

```python
grid = SimulationGrid(N=2048, backend="torch")
```

Everything downstream follows, and the recorded histories still come back as
host NumPy, so your plots and metrics are unchanged. Two things to know before
you reach for it: torch on Apple's MPS is `float32`, which puts a noise floor
of about `1e-4` under every error the benchmark reports; and at the 1D
transverse sizes this package uses, the GPU is often *slower* than the host.
See the [Simulation Grid](simulation_grid.md) guide and
`examples/full_backend_demo.py` for the details.

---

## Where to Go Next

* [Simulation Grid](simulation_grid.md) — every grid parameter, and the
  backend rules.
* [Samples Guide](samples_guide.md) — the six shipped samples, and how to
  write your own.
* [Results Guide](results_guide.md) — everything a `BenchmarkResult` can do.
* [Physics Simulation](physics_simulation.md) — what the four operators
  actually approximate.
