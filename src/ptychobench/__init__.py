"""
Benchmark forward-propagation operators for thick-sample ptychography.

The names below are the package's public surface: everything needed to define a
run, drive it, and read what came back. ``from ptychobench import
SimulationGrid, ZigBalls, run_benchmark`` is the intended way in, and it does
not require knowing which module holds what.

What is deliberately *not* re-exported is the numerical machinery -- the FFT
pair, the Arnoldi iteration, the splitting products, the band-limiting mask, the
error metrics. Those take arrays rather than grids and are usable on their own,
but they are building blocks rather than the interface, and hoisting them here
would put thirty-odd names in front of a reader looking for the five that
matter. They live in :mod:`ptychobench.numerics` and are imported from there.

``plotting`` and ``reporting`` are submodules rather than names, so
``from ptychobench import plotting`` reaches them; see
:doc:`the API reference <api_reference>` for what each draws or writes.
"""

import logging

from .benchmark import run_benchmark
from .grid import SimulationGrid
from .operators import (
    GroundTruthOperator,
    FeitFleckOperator,
    ForwardOperator,
    LinDudaOperator,
    ParaxialOperator,
    YevickThomsonOperator,
)
from .reporting import BenchmarkReport
from .results import BenchmarkResult
from .samples import (
    Apoferritin,
    Sample,
    SharpStraightWaveguides,
    StraightBalls,
    StraightWaveguides,
    ZigBalls,
    ZigWaveguides,
)

# A library must not configure logging for its host application. Attaching a
# NullHandler here suppresses the "No handlers could be found" fallback while
# leaving the choice of level, format and destination entirely to the caller.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # The domain
    "SimulationGrid",
    # Samples: the base class and the six shipped geometries
    "Sample",
    "Apoferritin",
    "StraightWaveguides",
    "SharpStraightWaveguides",
    "ZigWaveguides",
    "StraightBalls",
    "ZigBalls",
    # Operators: the base class, the reference, and the four approximations
    "ForwardOperator",
    "GroundTruthOperator",
    "ParaxialOperator",
    "FeitFleckOperator",
    "YevickThomsonOperator",
    "LinDudaOperator",
    # Running one, and what comes back
    "run_benchmark",
    "BenchmarkResult",
    "BenchmarkReport",
]
