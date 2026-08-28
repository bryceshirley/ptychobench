# Training Data Guide

In this section, you are exploring whether a neural network (Fourier Neural Operator) can learn the propagation step directly, and how does it measure up against the classical operators? This pipeline turn the benchmark simulation into a machine learning dataset and it builds on the [Simulation Grid](simulation_grid.md) and the
[samples](samples_guide.md), but does not go through `run_benchmark()` — it calls
`operator.step()` directly, so it can capture the wavefield at every step.

The four scripts run in order, and the output of each one is the input to the next. Run them from the repository root: every path below is relative to the working directory, so that is where the files appear.

| Script | Output | Purpose |
| --- | --- | --- |
| **`generate_data.py`** | `scripts/data/simulation_data.pt` | Randomises the sample parameters and records the inputs, exact targets and classical baseline at every propagation step. |
| **`inspect_data.py`** | a figure | Checks the generated data looks physically sensible. |
| **`train_fno.py`** | `fno_weights.pt` | Trains an FNO and measures how well it generalises. |
| **`compare_operators.py`** | a figure | Compares the trained FNO against the classical operators. |

`dataset.py` is not run directly; it holds the `PtychoDataset` class the other scripts import.

---

## 1. Generating a Dataset

`generate_data.py` walks through randomised sample parameters and, at every propagation step, records what the network needs to learn from: the wavefield going in, the sample it passes through, the exact answer, and what the classical Feit/Fleck operator predicts.

The tunable values sit at the top of `generate_data()`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`num_configs`** | `int` | `2` | How many times to draw a fresh set of random sample parameters. |
| **`grid_params`** | `dict` | see [Simulation Grid](simulation_grid.md) | The `SimulationGrid` settings shared by every sample. |
| `z_prop` | `float` | `100.0` | Total propagation distance in nm. |
| `Nz` | `int` | `2` | Number of z-steps. The step size is `dz = z_prop / Nz`. |
| `N` | `int` | `400` | Transverse pixels, and so the width of every saved row. |

Three sample types are generated for each configuration: `Apoferritin`, `StraightWaveguides` and `SharpStraightWaveguides`. Every sample gets its own random modulus; the two waveguide samples also get a random `num_cores` (the spatial frequency) and `core_width`, and `SharpStraightWaveguides` gets a random `blur`.

The core width is drawn as a fraction of the waveguide period rather than as an absolute width, so the guides stay separated however many cores are drawn.

**Usage:**

```bash
python src/ptychobench/generate_data.py
```

**Output:**

```text
scripts/data/simulation_data.pt
```

The dataset has `num_configs * 3 * Nz` rows. With the defaults that is 12.

### The Saved File

`simulation_data.pt` is a dictionary. Every tensor has shape `(rows, N)` and dtype `complex64`.

| Key | Type | Description |
| :--- | :--- | :--- |
| **`input_psi`** | `Tensor` | The wavefield entering the step, psi(z). A model input. |
| **`input_eps`** | `Tensor` | The refractive index perturbation epsilon at that z. A model input. |
| **`target_psi`** | `Tensor` | The ground truth psi(z + dz), from `ExactOperator`. The training target. |
| **`baseline_psi`** | `Tensor` | The classical prediction from `FeitFleckOperator`, stepped from the same psi so the residual is well defined. |
| **`grid_params`** | `dict` | The settings used, so the `SimulationGrid` can be rebuilt on load. |
| **`sample_names`** | `list[str]` | Which sample type each row came from. |
| **`config_ids`** | `list[int]` | Which draw of random parameters each row came from. |
| **`z_indices`** | `list[int]` | Which z-step each row is, as an index. |
| **`z_values`** | `list[float]` | Which z-step each row is, as a position in nm. |

The last four label every row, so nothing downstream has to work out what a row is from the order the generation loops happened to run in.

---

## 2. Checking the Dataset

Before training on it, it is worth looking at what came out. `inspect_data.py` prints a summary of every row and plots one configuration.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`CONFIG`** | `int` | `0` | Which of the `num_configs` configurations to plot. |
| **`Z_STEP`** | `int` | `0` | Which z-step of that configuration to plot. |

**Usage:**

```bash
python src/ptychobench/inspect_data.py
```

**Output:**

```text
scripts/results/dataset_check_config{CONFIG}_zstep{Z_STEP}.png
```

The figure has one row per sample type, showing the sample modulus, the wavefield magnitudes, and the phase. It is saved to `scripts/results/`, named after the configuration and z-step so different settings do not overwrite each other. Pass `--show` to open it in a window as well.

The printed table gives the relative error of the Feit/Fleck baseline against the exact solution for every row, which is a quick check that the parameters produce a meaningful spread of difficulty.

---

## 3. Loading the Data

`PtychoDataset` is a standard PyTorch `Dataset`. Neural networks cannot take complex numbers, so it splits the complex wavefields into real channels:

| Tensor | Shape | Channels |
| :--- | :--- | :--- |
| **input** | `(3, N)` | Real(psi), Imag(psi), epsilon |
| **target** | `(2, N)` | Real(psi_exact), Imag(psi_exact) |

Epsilon only needs one channel because its imaginary part is zero.

**Usage:**

```python
from torch.utils.data import DataLoader

from dataset import PtychoDataset

training_data = PtychoDataset()
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

train_features, train_targets = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")

```

**Output:** none. This one only reads the dataset; running the file directly
just prints the shape of one batch as a check.

---

## 4. Training an FNO

`train_fno.py` builds an FNO from the `neuraloperator` library, trains it on a single batch for many epochs, and measures how far that gets it on data it has never seen.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`N_MODES`** | `tuple` | `(64,)` | Fourier modes the operator keeps. One entry per spatial dimension, and the wavefields are 1D. |
| **`HIDDEN_CHANNELS`** | `int` | `64` | Width of the internal feature representation. |
| **`LEARNING_RATE`** | `float` | `1e-3` | The learning rate for Adam. |
| **`BATCH_SIZE`** | `int` | `8` | Size of the single batch the model trains on. |
| **`NUM_EPOCHS`** | `int` | `5000` | How many passes over that one batch. |
| `SPLIT_SEED` | `int` | `0` | Fixes which rows are held back, so "unseen" means the same thing every run. |

The loss is a relative L2 loss (`LpLoss`), which normalises by the size of the target so weak and strong perturbations count equally. A value of `0.1` reads as a 10% error.

`N_MODES` matters more than it looks. The FNO discards every Fourier mode above it, so it cannot represent detail finer than that. At `(16,)` only 28.5% of the spectral energy of the target is inside the cutoff, and the loss will not drop below about 0.13 however long it trains.

**Usage:**

```bash
python src/ptychobench/train_fno.py
```

**Output:**

```text
scripts/results/training_loss.png
scripts/results/generalisation.png
scripts/results/fno_weights.pt
```

A loss curve, a plot comparing predictions on trained and unseen data, and the trained weights. Runs are logged to Weights and Biases if you are signed in.

---

## 5. Comparing the Operators

`compare_operators.py` puts the trained FNO next to the classical operators on the same axes, and plots how far each one is from the ground truth.

**Usage:**

```bash
python src/ptychobench/compare_operators.py
```

The configuration and z-step can be given on the command line, which is easier than editing the file when you want several figures:

```bash
python src/ptychobench/compare_operators.py 0 1
```

**Output:**

```text
scripts/results/operator_comparison_config{CONFIG}_zstep{Z_STEP}.png
```

The figure has one row per sample type and four columns: the sample modulus, the wavefield magnitudes, the phases, and the error of each operator against the ground truth on a log scale. The title of each row says whether the FNO was trained on that sample, which changes the result completely.

This script loads the weights that `train_fno.py` saved rather than training a model of its own, so both report on the same model. Run `train_fno.py` first.
