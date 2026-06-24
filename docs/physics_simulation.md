# How the Physics Simulation Works

We need to model how a wave of light passes through an object (like a biological cell or a glass waveguide). This page explains the concepts powering the `ptychobench` simulations!

---

## 1. The Core Challenge (Coupled Physics)

As light travels forward, two distinct things happen at the exact same time
1. **Diffraction:** The light naturally spreads out as it travels through empty space.
2. **Scattering:** The physical object slows down and bends the light.

In the real world, these two effects are **coupled**—they happen simultaneously and affect each other constantly. 

Calculating the **Exact Operator** (the perfect mathematical rule that computes both of these at the exact same time) is possible in our 2D sandbox. However, in a full 3D simulation, calculating the exact physics requires too much computing power. Because of this, scientists invent approximations as shortcuts.

---

## 2. The Three Shortcuts We Are Testing

In this simulation, your goal is to test three of these shortcuts to see when they work well, and when they fail!

### 1. Paraxial Approximation (Q1)
* **The Idea:** This shortcut assumes that the light is mostly pointing straight ahead and isn't scattering very far to the sides. 
* **When it works:** It is valid for **small angles** (like a tightly focused laser pointer). 
* **When it fails:** If your light beam spreads out widely, or your sample causes the light to scatter at sharp angles, this shortcut will fail.

### 2. Feit/Fleck Approximation (Q2)
* **The Idea:** This shortcut artificially "splits" the physics: it calculates the light spreading in empty space first, and then calculates the light hitting the sample second.
* **When it works:** If you shine light through completely empty space (zero sample contrast) OR if the light travels perfectly straight (zero propagation angle), this shortcut is exact!
* **When it fails:** Because light spreading and light bending are supposed to be coupled, artificially splitting them into two steps creates a error. This error is small for low-contrast samples and small angles, but it grows as the sample gets more complex or the light spreads out more.

### 3. Lin/Duda Corrected Split-Step (Q3)
* **The Idea:** This shortcut acts as Q2, but with an extra piece of math attached to the end to clean up the mistake caused by splitting the physics.
* **When it works:** By correcting for the splitting error, Q3 provides significantly **higher accuracy** than Q2, allowing you to model complex objects and wider angles.

---

## 3. Why We Can Test These Shortcuts in 2D

Normally, it is impossible to know exactly how much "error" these shortcuts are making because calculating the Exact answer in 3D is too hard.

However, by restricting our simulation to **2D**, the problem becomes just small enough that a modern computer can calculate the correct coupled physics! This allows us to use the Exact answer to see how close the Q1, Q2, and Q3 shortcuts got to reality.