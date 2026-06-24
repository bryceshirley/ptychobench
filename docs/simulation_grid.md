# Simulation Grid

Before you can run an experiment, you need to build the laboratory. In `ptychobench`, the `SimulationGrid` is the digital sandbox where your physics simulation takes place. 

It defines the physical dimensions of the space, the resolution of the simulation, and the properties of the probe light beam entering the sample.

---

## The Core Requirement

When you create a grid, there is exactly **one** parameter you are required to provide. 

### Divergence Angle
This controls how much the initial beam of light spreads out as it travels forward. 


**Example Usage:**
```python
from ptychobench.grid import SimulationGrid

# Create a grid with a 15-degree light spread
grid = SimulationGrid(divergence_angle=15.0)

```

---

## Tuning the Sandbox (Default Parameters)

While you only *have* to provide the divergence angle, the Grid comes with several pre-set defaults that you can change to modify the physics or the quality of the simulation.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `lam` | `float` | `1.0` | **Wavelength:** The wavelength of the light. |
| `L` | `float` | `100.0` | **Sample Width:** The total physical width of the simulation space. |
| `z_prop` | `float` | `50.0` | **Propagation Length:** How far the light travels from start to finish. |
| `probe_width` | `float` | `5.0` | **Probe Size:** The initial thickness of the light beam before it spreads. |
| `N` | `int` | `128` | **Transverse Resolution:** How many horizontal pixels make up the sample. |
| `Nz` | `int` | `100` | **Longitudinal Steps:** How many "steps" the solver takes to cross the sample. |

**Example (Customizing the Grid):**

```python
# A high-resolution grid in a smaller physical space
grid = SimulationGrid(
    divergence_angle=10.0,
    L=50.0, 
    N=256, 
    Nz=200
)

```