# Experiments Summary

# Straight Waveguides
```
--- Grid & Physical Setup ---
divergence_angle: 80.0
lam:         1.0
L:           100.0
z_prop:      50.0
N (pixels):  128
Nz (steps):  100
probe_width: 5.0
dx:          7.812500e-01
dz:          5.000000e-01

--- Sample Setup ---
Sample:    StraightWaveguides
modulus     : 0.01
phase       : 0.0001
period      : 10.0
core_width  : 1.5
```
The intensity error is worse at the edges of the sample.

![intensity_error](/contributor_notes/Bryce_notes/intensity_error.png)

# Apoferritin Sample

## Modulus 1.0, Phase 0.01
```
Run Date/Time: 2026-06-30_16-21-24

--- Grid & Physical Setup ---
divergence_angle: 0.0
lam:         1.0
L:           100.0
z_prop:      50.0
N (pixels):  128
Nz (steps):  100
probe_width: 5.0
dx:          7.812500e-01
dz:          5.000000e-01

--- Sample Setup ---
Sample:    Apoferritin
modulus     : 1.0
phase       : 0.01
file_path   : /Users/bryce.shirley/Library/CloudStorage/OneDrive-ScienceandTechnologyFacilitiesCouncil/Documents/ptychobench/src/ptychobench/apoferritin_100Nz565Nx.npy
```
Feit-Fleck operator performs worse than the Paraxial operator.

RMSE of the intensity for Paraxial (Q1): 4.947258e+00
RMSE of the intensity for Feit/Fleck (Q2): 1.070490e+01
RMSE of the intensity for Lin/Duda (Q3): 1.059470e+00

## Modulus 0.1, Phase 0.001
```
Run Date/Time: 2026-06-30_16-28-26

--- Grid & Physical Setup ---
divergence_angle: 0.0
lam:         1.0
L:           100.0
z_prop:      50.0
N (pixels):  128
Nz (steps):  100
probe_width: 5.0
dx:          7.812500e-01
dz:          5.000000e-01

--- Sample Setup ---
Sample:    Apoferritin
modulus     : 0.1
phase       : 0.001
file_path   : /Users/bryce.shirley/Library/CloudStorage/OneDrive-ScienceandTechnologyFacilitiesCouncil/Documents/ptychobench/src/ptychobench/apoferritin_100Nz565Nx.npy
```
