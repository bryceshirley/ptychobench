"""
Numerical building blocks: arrays in, arrays out.

Everything here works on arrays alone. Nothing in this subpackage knows what a
:class:`~ptychobench.SimulationGrid` or a :class:`~ptychobench.Sample` is, and
that is the membership rule -- a module belongs here exactly when it can be used
without one, which is why the error metrics sit beside the FFT pair rather than
beside the benchmark they are usually called from.

The rule is worth stating because it is what makes the boundary checkable. The
modules above this one supply the physics: a grid, a sample, an operator built
from them. The modules here supply the mathematics those are assembled out of,
and they are deliberately reusable outside this package.

* :mod:`~ptychobench.numerics.transforms` -- the unitary FFT pair and
  ``fourier_multiply``.
* :mod:`~ptychobench.numerics.integrators` -- Arnoldi, and ``krylov_funm`` for
  matrix functions from a matrix-vector product alone.
* :mod:`~ptychobench.numerics.splitting` -- the Lie-Trotter product and the
  truncated series for the cross-term factor.
* :mod:`~ptychobench.numerics.apertures` -- band limiting. Opt-in; nothing
  applies one by default.
* :mod:`~ptychobench.numerics.metrics` -- RMSE and far-field error.

No names are re-exported here on purpose. These are imported from their own
modules, so a reader sees which of the five a helper came from at the import
line rather than having to look it up.
"""
