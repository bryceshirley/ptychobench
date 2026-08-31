# API Reference

A map of the package: which module holds what, and why the boundaries fall
where they do. The user guides cover how to *use* the main entry points; this
page is for when you want to reach past them.

Every public function and class has a full docstring, so `help(...)` in a
notebook is the authoritative source for signatures.

---

## The Package Root

Everything needed to define a run, drive it, and read the result is re-exported
from `ptychobench` itself, so the ordinary case never needs a module path:

```python
from ptychobench import SimulationGrid, ZigBalls, run_benchmark
from ptychobench import FeitFleckOperator, LinDudaOperator
```

Seventeen names in all: `SimulationGrid`; `Sample` and the six shipped
geometries; `ForwardOperator`, `GroundTruthOperator` and the four
approximations; `run_benchmark`, `BenchmarkResult` and `BenchmarkReport`.

The numerical building blocks are **not** hoisted here — they take arrays
rather than grids, and are machinery rather than interface. Import those from
their modules, listed below.

`import ptychobench` deliberately does **not** import Matplotlib: every `plot_*`
method defers the import to call time, so a headless script that only wants
arrays back does not pay for the plotting stack.

---

## The Modules

The package is one flat level plus a single subpackage. The dividing line is
whether a module needs a `SimulationGrid`: the ones that do are the physics and
stay at the top; the ones that work on bare arrays live in `numerics`.

**Top level — needs a grid, a sample, or a result:**

| Module | Holds |
| :--- | :--- |
| `ptychobench.grid` | `SimulationGrid` — the domain, its derived arrays, and the backend choice. |
| `ptychobench.samples` | `Sample` and the six shipped samples. Geometry only. |
| `ptychobench.operators` | `ForwardOperator` and the five propagation operators. |
| `ptychobench.benchmark` | `run_benchmark` — the propagation loop. |
| `ptychobench.results` | `BenchmarkResult` — what a run produced. |
| `ptychobench.plotting` | Every figure the package draws. |
| `ptychobench.reporting` | `BenchmarkReport` — a result written to disk. |
| `ptychobench.utils` | Backend resolution, host transfer, the `Array` protocol. |

**`ptychobench.numerics` — arrays in, arrays out:**

| Module | Holds |
| :--- | :--- |
| `ptychobench.numerics.transforms` | Unitary FFT pair and `fourier_multiply`. |
| `ptychobench.numerics.integrators` | `krylov_funm` — Arnoldi for matrix functions. |
| `ptychobench.numerics.splitting` | `lie_trotter` and `taylor_exponential` — operator splitting. |
| `ptychobench.numerics.apertures` | `antialias_mask` — band limiting. Opt-in; see below. |
| `ptychobench.numerics.metrics` | RMSE and far-field error functions. |

Nothing in `numerics` imports anything above it except `utils`, and a test
enforces that. `metrics` is there because `calculate_rmse` takes two arrays and
nothing else — grouping by what a module *needs* rather than by when it is
usually called is what keeps the rule a rule instead of a habit.

---

## Full Benchmark Interface

```python
from ptychobench import (
    SimulationGrid,
    ZigBalls,
    ParaxialOperator,
    FeitFleckOperator,
    run_benchmark,
)

result = run_benchmark(
    SimulationGrid(N=512, L=40.0, z_prop=20.0),
    ZigBalls(modulus=0.08),
    [ParaxialOperator, FeitFleckOperator],
    integrator="krylov",
)
```

`run_benchmark(grid, sample, operators, integrator="direct") -> BenchmarkResult`
is the whole entry point. `operators` takes a single class or a sequence;
`GroundTruthOperator` is appended if you left it out, since it is the reference
every error is measured against.

See the [Results Guide](results_guide.md) for what comes back.

---

## Operators

Five classes, all subclasses of `ForwardOperator`, all constructed the same
way: `OpClass(grid, integrator="direct")`.

| Class | Generator | Split-step form |
| :--- | :--- | :--- |
| `GroundTruthOperator` | $\sqrt{1+\mu+\varepsilon}-1$ | none |
| `ParaxialOperator` | $(\varepsilon+\mu)/2$ | separable |
| `FeitFleckOperator` | $L+N$ | separable |
| `YevickThomsonOperator` | $H_2 - (\varepsilon\mu+\mu\varepsilon)/8$ | cross term |
| `LinDudaOperator` | $H_2 - (LN+NL)/2$ | cross term |

`integrator="split-step"` needs one of the two right-hand entries. "Separable"
means two diagonal pieces and nothing else, so the step is a product of two
elementwise factors; "cross term" means a third factor is needed, at the cost
of extra transforms and a step that is no longer exactly unitary — see
[Technical Theory §4.2](technical.md). Only `GroundTruthOperator` has neither,
and asking it for `"split-step"` raises `NotImplementedError` at construction.

The interface each one implements:

* **`step(E, psi)`** — advance the field by one `dz`. `E` is the permittivity,
  either as a length-`N` vector or as the `N x N` diagonal matrix; the path
  that needs the other form converts for you. Returns a field in `psi`'s
  namespace, dtype and device.
* **`construct_operator(E)`** — the generator as a dense `N x N` matrix. Host
  NumPy. What `"direct"` exponentiates.
* **`apply(eps, psi)`** — the matrix-free action, equal to
  `construct_operator(diag(eps)) @ psi` and asserted to be so in the test
  suite. What `"krylov"` builds its subspace from. `GroundTruthOperator` raises
  here: a matrix square root has no elementwise action, so it takes a different
  Krylov route.
* **`split_step_parts(eps)`** — the Fourier-diagonal and real-space-diagonal
  halves. What `"split-step"` exponentiates separately.
* **`cross_term_action(eps, psi)`** — for an operator with a cross term, the
  part `split_step_parts` leaves out, so that `apply` is the sum of the two.
  Written once and used by both paths, so a sign slip cannot be right in one
  and wrong in the other. `cross_term_order` sets how many series terms are
  kept for its exponential; the default is 4, and the operator warns when that
  stops being enough.
* **`display_name`** — the plot label, declared on the class. A subclass that
  declares none fails at import rather than showing a blank legend entry.

To add an operator, subclass `ForwardOperator`, declare a `display_name`, and
implement `construct_operator`. Add `apply` if you want Krylov. For
`"split-step"`, add `split_step_parts` and either `separable = True` or
`cross_term = True` plus `cross_term_action`.

---

## Integrators, Splitting and Transforms

The numerical building blocks the operators are assembled from. They take
arrays, not grids, so they are usable on their own.

```python
from ptychobench.numerics.integrators import krylov_funm
from ptychobench.numerics.splitting import lie_trotter, taylor_exponential
from ptychobench.numerics.transforms import fft, ifft, fourier_multiply
```

* `krylov_funm(matvec, v, f, *, max_iter=30, tol=1e-8)` — evaluate `f(A) @ v`
  from the action of `A` alone, by projecting onto an Arnoldi subspace and
  applying `f` to the small projection.
* `lie_trotter(psi, kinetic_symbol, potential, step, *, aperture=None)` — the
  first-order splitting `exp(step * L) exp(step * N)`. The only one the package
  offers.
* `taylor_exponential(apply_generator, psi, order)` — `exp(Omega) @ psi` from
  the action of `Omega` alone, as a truncated series. The third factor of an H3
  or H4 step. Unlike `lie_trotter` this is not unitary, and
  `warn_if_series_truncated` is the diagnostic for when `order` has stopped
  being enough.
* `fourier_multiply(psi, symbol, axes=None)` — apply a Fourier-diagonal
  operator, i.e. `ifft(symbol * fft(psi))`.

Two things to know about the transforms. The FFT pair is **unitary** — the
`1/sqrt(N)` is split between the forward and inverse transforms, so Parseval
holds — which is not SciPy's convention. And they transform **every axis** by
default, because nothing here carries a batch axis: a rank-2 field is a 2D
transverse field, not a stack of 1D ones. Pass `axes` explicitly if you mean
otherwise.

---

## Band Limiting

`antialias_mask(kx, cutoff=2/3, edge="binary", taper=0.0)` builds a
band-limiting mask over the Fourier axis, which you can hand to `lie_trotter`
as `aperture=`. With `edge="tapered"` the raised-cosine roll-off lies *inside*
the cutoff, spanning `(cutoff - taper, cutoff)`, so nothing above the cutoff is
transmitted by either edge.

Watch the units on `taper`: here it is a fraction of the Nyquist wavenumber,
whereas abTEM measures its taper as a fraction of `1/dx`, which is *twice*
Nyquist. abTEM's default `0.01` is `taper=0.02` here. The cutoff conventions do
agree.

**Nothing in the package applies an aperture by default**, because one does not
commute with the real-space potential and so changes the operator being
integrated. `run_benchmark` therefore has no `aperture=` argument, and a
band-limited run means driving the propagation loop yourself:
`FeitFleckOperator.split_step_parts(eps)` hands back the two diagonals, and
`lie_trotter` takes them plus the mask.

[Technical Theory §4.4](technical.md) covers what band limiting does and does
not protect, including why masking the transmission function as a multislice
code does diverges here.

---

## Metrics

```python
from ptychobench.numerics.metrics import (
    calculate_rmse,
    calculate_max_error,
    calculate_rmse_intensity,
    calculate_max_intensity_error,
    calculate_farfield_wave,
)
```

`run_benchmark` calls these for you and puts the results on the
`BenchmarkResult`; they are exposed for writing your own comparisons.
`calculate_farfield_wave(exit_wave, mode="intensity")` is the propagation to
the detector — an unnormalised, shifted FFT — with `mode` selecting
`"intensity"`, `"magnitude"` or `"phase"`.

`calculate_rmse` and `calculate_max_error` compare two arrays as they are. The
two `*_intensity` forms transform both arguments first and then delegate, which
is convenient for a single number and wasteful for several: for more than one
metric off the same pair, call `calculate_farfield_wave` once per array and
pass the intensities to the plain forms.

---

## Backends

```python
from ptychobench.utils import resolve_backend, to_numpy, to_device
```

* `resolve_backend(name, device=None)` returns a `Backend`: the array
  namespace, the resolved device, and the real and complex dtypes for it. This
  is what `SimulationGrid` uses internally, and the honest answer to "what
  precision will I get" — which depends on the *device*, not just the library.
* `to_numpy(array)` brings any backend's array to the host, detaching a
  gradient-tracking tensor on the way. This is the conversion at every boundary
  where host-only code begins: Matplotlib, SciPy, and the recorded histories.
* `to_device(array, device)` moves an array, and is a no-op where placement has
  no meaning.

`utils` also defines `Array`, a structural `Protocol` describing the minimum a
value needs to be usable as an array here. It is used on **parameters**, as a
deliberately loose lower bound; functions that *return* a backend array are
annotated `Any`, because the concrete type depends on the backend.

---

## Reporting and Plotting

`ptychobench.plotting` is the only module that imports Matplotlib at module
scope, and no function there calls `show()` — figures are returned, and what
happens to them is the caller's decision. The `plot_*` methods on
`BenchmarkResult` and on `Sample` are one-line delegators into it, and they
import `plotting` *inside* the method, which is what keeps `import ptychobench`
free of Matplotlib. The test suite pins that.

```python
from ptychobench import plotting, reporting

# Taking a result
fig = plotting.plot_sample(result)
fig_phase, fig_amp = plotting.plot_wavefields(result)
fig_phase, fig_amp = plotting.plot_evolution_1D(result, x=0.0)
fig, widgets = plotting.plot_evolution_slider(result, x=0.0)
fig = plotting.plot_farfield_error(result, log_scale=False, mode="intensity",
                                   operators=None)

# Taking a sample and a grid
fig = plotting.plot_sample_profile(sample, grid)
fig = plotting.plot_sample_cross_section_modulus(sample, grid, z=0.0, fourier=False)
fig = plotting.plot_sample_cross_section_phase(sample, grid, z=0.0)

# Writing a whole run to disk
path = reporting.BenchmarkReport(save_dir="results").write(result)
```

`BenchmarkReport(save_dir).write(result)` is exactly what
`result.generate_report(save_dir)` calls. It creates a timestamped
subdirectory, writes both text summaries and every figure, and returns the
`Path`.
