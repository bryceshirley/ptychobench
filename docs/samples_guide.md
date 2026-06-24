# Samples Guide

In this simulation, you are exploring how light travels through different objects. Think of these "Samples" as the items you are placing under the microscope. Each one will create a different pattern of light when you run the simulation.

---

## 1. Apoferritin

A simulation of a experimental biological sample derived from a 2D cross-section array. This is ideal for testing realistic, complex diffraction patterns. Apoferritin is a protein that stores iron in cells, and its structure is well-studied in the field of structural biology.

![Apoferritin](images/Apoferritin.png)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`modulus`** | `float` | `0.08` | The strength (amplitude) of the scattering potential. |
| `phase` | `float` | `modulus * 0.5` | Phase perturbation; derived directly from the normalized structure. |

**Usage:**

```python
from ptychobench.samples import Apoferritin

sample = Apoferritin(modulus=0.8)

```

---

## 2. StraightWaveguides (The Light Tunnels)

A periodic array of straight waveguides, useful for studying how light traps within defined channels without vertical variation.

![StraightWaveguides](images/StraightWaveguides.png)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`modulus`** | `float` | `0.08` | The strength of the waveguides. |
| `phase` | `float` | `modulus * 1e-2` | Phase perturbation. |
| `period` | `float` | `10.0` | Horizontal spacing between waveguide centers. |
| `core_width` | `float` | `1.5` | The width of the Gaussian waveguide core. |

**Usage:**

```python
from ptychobench.samples import StraightWaveguides

sample = StraightWaveguides(modulus=0.8)

```

---

## 3. ZigWaveguides (The Bending Tunnels)

Similar to `StraightWaveguides`, but with an added "wiggle" along the propagation axis.

![ZigWaveguides](images/ZigWaveguides.png)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`modulus`** | `float` | `0.08` | The strength of the waveguides. |
| `wiggle_amp` | `float` | `2.0` | The amplitude of the z-axis curvature. |
| `period` | `float` | `10.0` | Horizontal spacing. |

**Example Usage:**

```python
from ptychobench.samples import ZigWaveguides

sample = ZigWaveguides(modulus=0.8)

```

---

## 4. StraightBalls (The Dot Grid)

A sequence of discrete, localized spherical features arranged in vertical lines. Use this to study scattering from round objects lined up in a row.

![StraightBalls](images/StraightBalls.png)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`modulus`** | `float` | `0.08` | Scattering strength. |
| `num_balls` | `int` | `15` | Number of spherical features along the z-axis. |
| `ball_radius` | `float` | `0.8` | The size of the spherical feature. |
| `x_period` | `float` | `5.0` | Horizontal spacing. |

**Example Usage:**

```python
from ptychobench.samples import StraightBalls

sample = StraightBalls(modulus=0.8)

```

---

## 5. ZigBalls

Spherical features arranged in a sharp, piecewise zig-zag pattern.

![ZigBalls](images/ZigBalls.png)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| **`modulus`** | `float` | `0.08` | Scattering strength. |
| `x_amplitude` | `float` | `3.0` | How far the balls shift horizontally in the zig-zag. |
| `num_balls` | `int` | `25` | Total number of spheres. |

**Example Usage:**

```python
from ptychobench.samples import ZigBalls

sample = ZigBalls(modulus=0.8)
```