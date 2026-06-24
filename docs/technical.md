### **Mathematical Framework**

In Ptychography, diffraction through a sample can be modeled by the forward Helmholtz equation:

$$\frac{\partial \psi}{\partial z} + i k_0 Q \psi = 0$$

where $Q$ is the pseudo-differential operator that depends on the sample's refractive index distribution perturbation ($\epsilon$) and the kinetic operator ($\mu$). The exact forward operator is defined as:

$$Q = \sqrt{I + \epsilon + \mu} - I$$

The exponential solution to the forward propagation over a step size $dz$ is given by:

$$\psi(z + dz) = \exp(i k_0 Q \, dz) \psi(z)$$

### **Propagation Approximations**

In 3D, computing the exact operator $Q$ is computationally expensive. Because of this, various approximations are used in practice which allow for efficient numerical methods. 

However, by restricting our simulation to 2D, we can directly evaluate the exact operator using matrix decomposition and compare it against these standard approximations:

* **Q1 (Paraxial Approximation):** Assumes small scattering angles.

$$Q_1 = \frac{1}{2}(\epsilon + \mu)$$


* **Q2 (Feit/Fleck Split-Step):** Separates the kinetic and environmental components.

$$Q_2 = \mathcal{L} + \mathcal{N}$$

*(where $\mathcal{L} = \sqrt{I + \mu} - I$ and $\mathcal{N} = \sqrt{I + \epsilon} - I$)*
* **Q3 (Lin/Duda):** Improves upon Feit/Fleck by explicitly subtracting the first-order commutation error.

$$Q_3 = Q_2 - \frac{1}{2}(\mathcal{LN} + \mathcal{NL})$$


*This framework allows us to rigorously test the accuracy of these approximations across different sample geometries, refractive index perturbations, and beam divergence angles.*

---

### **Numerical Methods**

This codebase relies on high-order SciPy solvers

* **Matrix Exponentials:** The code uses `scipy.linalg.expm`, which calculates the matrix exponential using a high-order Padé approximation algorithm.
> Awad H. Al-Mohy and Nicholas J. Higham (2009). *"A New Scaling and Squaring Algorithm for the Matrix Exponential"*, SIAM J. Matrix Anal. Appl. 31(3):970-989. DOI: 10.1137/09074721X


* **Matrix Square Roots:** The code utilizes `scipy.linalg.sqrtm`, which employs the Schur decomposition method to compute the square root of complex matrices.
> Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013). *"Blocked Schur Algorithms for Computing the Matrix Square Root"*, Lecture Notes in Computer Science, 7782, pp. 171-182. DOI: 10.1016/0024-3795(87)90118-2