# How the Physics Simulation Works

This page explains the ideas behind the `ptychobench` simulations in plain
terms. For the derivations, the operator algebra and the numerical methods,
see [Technical Theory](technical.md).

---

## 1. The Core Challenge (Coupled Physics)

As light travels forward, two distinct things happen at the exact same time:

1. **Diffraction:** the light naturally spreads out as it travels through empty
   space.
2. **Scattering:** the physical object slows down and bends the light.

In the real world these two effects are **coupled** — they happen
simultaneously and affect each other constantly. Computing the exact rule for
both at once is possible in our 2D sandbox but far too expensive in 3D, so
practitioners reach for approximations instead. This package measures what
those approximations cost.

Everything in the package is written in terms of one generator, `H`. The field
obeys

$$\frac{d\psi}{dz} = i k_0 H \psi,$$

so one step forward is $\exp(i k_0 \Delta z\, H)$, and the *only* thing the
operators disagree about is what `H` is. Two ingredients go into it:

* $\mu$ — the **kinetic** term, which is diffraction. Simple in Fourier space,
  because each spatial frequency just picks up a phase.
* $\varepsilon$ — the **potential** term, which is the sample. Simple in real
  space, because each position just picks up a phase.

The exact answer is $H = \sqrt{1 + \mu + \varepsilon} - 1$. The difficulty is
entirely that $\mu$ and $\varepsilon$ are simple in *different* places, and do
not commute.

---

## 2. The Four Shortcuts We Are Testing

### H1 — Paraxial Approximation

$H_1 = (\varepsilon + \mu)/2$

* **The idea:** truncate the square root at first order in both terms.
  Equivalent to assuming the light is mostly pointing straight ahead and is not
  scattering far to the sides.
* **When it works:** small angles — a tightly focused laser pointer.
* **When it fails:** anywhere else. It is the crudest of the four, and it is
  wrong even in *empty space*: at $\varepsilon = 0$ it gives $\mu/2$ where the
  right answer is $\sqrt{1+\mu} - 1$.

### H2 — Feit/Fleck

$H_2 = L + N$, with $L = \sqrt{1+\mu} - 1$ and $N = \sqrt{1+\varepsilon} - 1$

* **The idea:** keep each term exactly, but drop every *cross* term between
  them — calculate the spreading in empty space and the effect of the sample
  separately, then add them.
* **When it works:** exactly, if either ingredient vanishes. Empty space (zero
  sample contrast) or perfectly straight light (zero propagation angle) both
  come out right.
* **When it fails:** because spreading and bending really are coupled,
  separating them costs you the first cross term,
  $-(\varepsilon\mu + \mu\varepsilon)/8$. That error is small for low-contrast
  samples and small angles, and grows as either increases.
* **Bonus:** this is the one operator that is *by construction* one
  Fourier-diagonal piece plus one real-space-diagonal piece, which is what
  makes the fast split-step method possible at all.

### H3 — Yevick/Thomson

$H_3 = H_2 - (\varepsilon\mu + \mu\varepsilon)/8$

* **The idea:** H2 with the leading cross term put back.
* **Why it is symmetrised:** the unsymmetrised $-\varepsilon\mu/4$ matches the
  same Taylor expansion, but quietly stops conserving energy. Writing both
  orderings keeps the generator self-adjoint, so the step stays unitary.

### H4 — Lin/Duda

$H_4 = H_2 - (LN + NL)/2$

* **The idea:** the same correction, but built from the resummed factors $L$
  and $N$ rather than from the raw $\mu$ and $\varepsilon$.
* **When it wins:** same order of accuracy as H3, but better behaved at wide
  angles, because $L$ stays bounded where $\mu$ does not.

H3 and H4 both pay for their accuracy in flexibility. Their cross terms are
products *across* the two bases, so they are diagonal in neither and no product
of two elementwise factors can represent them. They still reach the fast
method, by carrying the cross term in a *third* factor — the trick Lin and Duda
introduce — which costs extra transforms per step and leaves the field drifting
slightly in norm where H1 and H2 do not.

### What the benchmark actually measures

Expand each generator in the size of the perturbation `s`, and the error in a
single step scales as a power of it. Measured on this package's own test suite:

| Operator | Measured order |
| :--- | :--- |
| H1 Paraxial | 2.02 |
| H2 Feit/Fleck | 2.02 |
| H3 Yevick/Thomson | 3.04 |
| H4 Lin/Duda | 3.04 |

The two pairs are indistinguishable in *order*, which is the point: H3 and H4
buy you a whole extra power of `s`, and the choice between them is about
behaviour at wide angles rather than about convergence rate.

---

## 3. Why We Can Test These Shortcuts in 2D

Normally it is impossible to know how much error these shortcuts make, because
calculating the exact answer in 3D is far too hard. A genuine 3D problem with
an `Nx x Ny` transverse field needs an `(Nx·Ny) x (Nx·Ny)` operator — for a
modest 200 x 200 field, a 40000 x 40000 matrix, about 12.8 GB in complex128.

By restricting the simulation to **2D**, though, the transverse problem becomes
one-dimensional — an `N`-element vector rather than an `N x N` image. The exact
generator is then an `N x N` matrix whose square root and exponential a modern
computer can simply *form*, at $O(N^3)$. That is what `GroundTruthOperator` does, and
it is why every error in this package is measured against a genuine reference
rather than against a finer approximation.

---

## 4. Evaluating the Exponential

Choosing `H` is one decision. *How to exponentiate it* is a separate,
orthogonal one, and `run_benchmark` takes it as its `integrator` argument:
`"direct"` forms the dense matrix, `"krylov"` projects onto a small subspace,
and `"split-step"` exponentiates the halves separately. They trade cost against
accuracy and against which operators they support — see
[Technical Theory §4.3](technical.md) for the comparison.

The ground-truth operator has no split-step decomposition and never will. Ask for one
and you get a clear refusal rather than silently dropped terms; `run_benchmark`
falls back to Krylov for it and logs that it did, so a split-step run still
measures its RMSEs against a reference evaluated a different way.
