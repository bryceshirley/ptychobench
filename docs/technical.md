# Wave Propagation Framework: Mathematical Background

## 1. Governing Equations: Helmholtz Equation to Forward Propagation Equation

The stationary state of monochromatic light propagation through a medium with refractive index $n$ is governed by the Helmholtz equation:

$$\nabla^2 u + k_0^2 n^2 u = 0$$

Solving this exactly in 3D is computationally expensive. To simplify, we define the transverse operator $\mathcal{Q}$ as:

$$\mathcal{Q} = \sqrt{1 + \mu + \epsilon}$$

where $\epsilon = n^2 - 1$ is the refractive index perturbation and $\mu = k_0^{-2}\nabla_{\perp}^2$ is the kinetic operator. 

Using this, we can factor the Helmholtz equation:

$$\left(\frac{\partial}{\partial z} + i k_0 \mathcal{Q}\right)\left(\frac{\partial}{\partial z} - i k_0 \mathcal{Q}\right) u  - i k_0 \left[\mathcal{Q},\frac{\partial}{\partial z}\right]u = 0$$

Assuming that backscattering and the commutator term are negligible, we can reduce the Helmholtz equation to a first-order differential equation in $z$:

$$\frac{\partial u}{\partial z} = i k_0 \mathcal{Q} u$$

This is the forward Helmholtz propagation equation. To remove the fast-oscillating phase term of the carrier wave, we consider the envelope equation $u = \exp(i k_0 z) \psi$. 

$$\frac{\partial \psi}{\partial z} = i k_0 (\mathcal{Q}-1) \psi$$

## 2. The Exponential Propagation Operator

The forward propagation equation can be expressed in terms of the transverse operator $\mathcal{H}$ as:

$$\frac{\partial \psi}{\partial z} = i k_0 \mathcal{H} \psi$$

where $\mathcal{H} = \mathcal{Q} - 1$ is the transverse operator.

The exact exponential solution to propagate the forward wave over a step size $dz$ is given by:

$$\psi(z + dz) = \exp(i k_0 \mathcal{H} dz) \psi(z)$$


## 3. Propagation Approximations

In 3D, computing the $\mathcal{H}$ is computationally unfeasible. However, by restricting our simulation to 2D, we can directly evaluate the operator and rigorously benchmark it against standard approximations that are required in practice.

The exact operator $\mathcal{H}$ can be expanded via a Taylor series as:

$$\mathcal{H}  = \frac{1}{2}(\epsilon + \mu) - \frac{1}{8}(\epsilon^2 + \epsilon\mu + \mu\epsilon + \mu^2) + \dots$$

The following implemented approximations represent different truncations and corrections of this series:

* **$\mathcal{H}_1$ (Paraxial Approximation):** Assumes small scattering angles. 
    $$\mathcal{H}_1 = \frac{1}{2}(\epsilon + \mu)$$
* **$\mathcal{H}_2$ (Feit/Fleck Split-Step):** Separates the kinetic and environmental components. It is exact when either the potential or kinetic operator is zero.
    $$\mathcal{H}_2 = \mathcal{L} + \mathcal{N}$$
    *(Note: $\mathcal{N} = \sqrt{1 + \epsilon} - 1$ is the potential operator, $\mathcal{L} = \sqrt{1 + \mu} - 1$ is the exact free-space kinetic operator).*
* **$\mathcal{H}_3$ (Yevick/Thomson):** Improves upon Feit/Fleck by explicitly subtracting the second-order cross terms.
    $$\mathcal{H}_3  = \mathcal{H}_2 - \frac{1}{8}(\epsilon\mu + \mu\epsilon)$$
* **$\mathcal{H}_4$ (Lin/Duda):** An alternative improvement upon Feit/Fleck that subtracts the second-order commutation error using the separated operators, offering better numerical stability.
    $$\mathcal{H}_4 = \mathcal{H}_2  - \frac{1}{2}(\mathcal{LN} + \mathcal{NL})$$

* Future work will utilize operator learning to improve upon the Feit/Fleck approximation, which is currently the most widely used in practice.

### Error Analysis via Taylor Expansion

The table below illustrates how the higher-order approximations successfully recover the missing cross-terms ($\epsilon\mu$ and $\mu\epsilon$) present in the exact operator up to $\mathcal{O}(x^2)$.

| Operator | Method | Taylor Expansion | Order of Accuracy |
| :--- | :--- | :--- | :--- |
| $\mathcal{H}_{Exact}$ | Ground Truth | $\frac{1}{2}(\epsilon + \mu) - \frac{1}{8}(\epsilon^2 + \epsilon\mu + \mu\epsilon + \mu^2) + \dots$ | Infinite |
| $\mathcal{H}_1$ | Paraxial | $\frac{1}{2}(\epsilon + \mu)$ | First Order |
| $\mathcal{H}_2$ | Feit/Fleck | $\frac{1}{2}(\epsilon + \mu) - \frac{1}{8}(\epsilon^2 + \mu^2) + \dots$ | First Order |
| $\mathcal{H}_3$ | Yevick/Thomson | $\frac{1}{2}(\epsilon + \mu) - \frac{1}{8}(\epsilon^2 + \epsilon\mu + \mu\epsilon + \mu^2) + \dots$ | Second Order  |
| $\mathcal{H}_4$ | Lin/Duda | $\frac{1}{2}(\epsilon + \mu) - \frac{1}{8}(\epsilon^2 + \epsilon\mu + \mu\epsilon + \mu^2) + \dots$ | Second Order |

> *Note: This framework allows us to rigorously test the accuracy of these approximations across different sample geometries, refractive index perturbations, and beam divergence angles.*

---

## 4. Numerical Methods

This codebase relies on high-order SciPy solvers for rigorous matrix evaluations:

* **Matrix Exponentials:** The code uses `scipy.linalg.expm`, which calculates the matrix exponential using a high-order Padé approximation algorithm.
* **Matrix Square Roots:** The code utilizes `scipy.linalg.sqrtm`, which employs the Schur decomposition method to compute the square root of complex matrices.

---

## 5. Operator Definitions

To compute the kinetic components numerically, this codebase utilizes the Angular Spectrum Method, evaluating the derivatives in the Fourier domain. 

* **Kinetic Operator ($\mu$):**
    $$\mu = \mathcal{F}^{-1} \left[ - k_0^{-2}k_x^2 \right] \mathcal{F}$$
* **Angular Spectrum Operator ($\mathcal{L}$):** $$\mathcal{L} = \mathcal{F}^{-1} \left[ \sqrt{1 - k_0^{-2}k_x^2} - 1 \right] \mathcal{F}$$

where $\mathcal{F}$ is the Fourier transform operator, and $k_x$ is the transverse spatial frequency. This enforces periodic boundary conditions.

---

## 6. References

> Al-Mohy, A. H., & Higham, N. J. (2009). A New Scaling and Squaring Algorithm for the Matrix Exponential. *SIAM Journal on Matrix Analysis and Applications*, 31(3), 970-989. DOI: [10.1137/09074721X](https://doi.org/10.1137/09074721X)

> Deadman, E., Higham, N. J., & Ralha, R. (2013). Blocked Schur Algorithms for Computing the Matrix Square Root. *Lecture Notes in Computer Science*, 7782, 171-182. DOI: [10.1016/0024-3795(87)90118-2](https://doi.org/10.1016/0024-3795(87)90118-2)

> Feit, M. D., & Fleck, J. A. (1978). Light propagation in graded-index optical fibers. *Applied Optics*, 17(24), 3990-3998. DOI: [10.1364/AO.17.003990](https://doi.org/10.1364/AO.17.003990)

> Lin, Y.-T., & Duda, T. F. (2012). A higher-order split-step Fourier parabolic-equation sound propagation solution scheme. *The Journal of the Acoustical Society of America*, 132(2), EL61-EL67. DOI: [10.1121/1.4730328](https://doi.org/10.1121/1.4730328)

> Yevick, D., & Thomson, D. J. (1994). Split-step/finite-difference and split-step/Lanczos algorithms for solving alternative higher-order parabolic equations. *The Journal of the Acoustical Society of America*, 96(1), 396-405. DOI: [10.1121/1.410490](https://doi.org/10.１１２１/１．４１０４９０)
