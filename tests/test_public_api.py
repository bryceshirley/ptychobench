"""Tests for the package's top-level namespace.

These pin what ``import ptychobench`` offers, independently of which module any
name happens to live in. That independence is the point: the module layout is
free to change -- and did, when the array-only modules moved into
``ptychobench.numerics`` -- while these tests keep saying the same thing.
"""

import ast
from pathlib import Path

import pytest

import ptychobench

# --- The curated surface ---

#: Every name the package promises at the top level. Spelled out here rather
#: than read off `__all__`, so that a name being dropped from the export list
#: fails a test instead of quietly agreeing with itself.
PUBLIC_NAMES = [
    "SimulationGrid",
    "Sample",
    "Apoferritin",
    "StraightWaveguides",
    "SharpStraightWaveguides",
    "ZigWaveguides",
    "StraightBalls",
    "ZigBalls",
    "ForwardOperator",
    "GroundTruthOperator",
    "ParaxialOperator",
    "FeitFleckOperator",
    "YevickThomsonOperator",
    "LinDudaOperator",
    "run_benchmark",
    "BenchmarkResult",
    "BenchmarkReport",
]

#: Names that must *not* appear at the root. One per `numerics` module, since
#: the thing being guarded against is a whole module's worth of helpers being
#: hoisted for convenience, not an individual name slipping through.
WITHHELD_NAMES = [
    "fourier_multiply",  # numerics.transforms
    "krylov_funm",  # numerics.integrators
    "lie_trotter",  # numerics.splitting
    "antialias_mask",  # numerics.apertures
    "calculate_rmse",  # numerics.metrics
]

#: Every module in the subpackage, resolved at collection time so a new one is
#: covered by the dependency rule below without anyone remembering to add it.
NUMERICS_SOURCES = sorted((Path(ptychobench.__file__).parent / "numerics").glob("*.py"))


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_the_public_name_resolves_at_the_package_root(name):
    """`from ptychobench import X` works for every advertised name."""
    assert hasattr(ptychobench, name)


def test_the_export_list_is_exactly_the_public_names():
    """`__all__` lists those names and no others, so `import *` agrees.

    Distinct from the test above: a name can resolve as an attribute while
    being absent from `__all__`, which is how a re-export gets added without
    being documented.
    """
    assert sorted(ptychobench.__all__) == sorted(PUBLIC_NAMES)


# --- What is deliberately absent ---


@pytest.mark.parametrize("name", WITHHELD_NAMES)
def test_the_numerical_building_blocks_are_not_hoisted_to_the_root(name):
    """The array-only helpers stay in `ptychobench.numerics`.

    They are usable on their own, but they are machinery rather than interface:
    hoisting them would put thirty-odd names in front of a reader looking for
    the handful that run a benchmark. This is the guard on that decision -- it
    should fail loudly if someone adds a convenience re-export, so the choice is
    made again on purpose rather than eroded one module at a time.
    """
    assert not hasattr(ptychobench, name)


def test_importing_the_package_does_not_pull_in_matplotlib():
    """`import ptychobench` stays free of the plotting stack.

    Re-exporting `BenchmarkResult` and `BenchmarkReport` from the root is what
    put this at risk: both had a module-scope `plotting` import, so hoisting
    them would have made every headless script that only wants an array pay for
    Matplotlib. Both now defer it per method, matching `Sample`'s delegators.

    A subprocess because the assertion is about a *fresh* interpreter: by the
    time this file runs, half the suite has imported pyplot already.
    """
    import subprocess
    import sys

    probe = "import sys, ptychobench; sys.exit('matplotlib' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", probe]).returncode == 0


# --- The subpackage boundary ---


@pytest.mark.parametrize("source_file", NUMERICS_SOURCES, ids=lambda p: p.stem)
def test_the_numerics_module_depends_on_nothing_above_it_but_utils(source_file):
    """Nothing in `numerics` reaches back up the package except for `utils`.

    This is the membership rule made checkable. `numerics` holds the modules
    that work on arrays alone -- a module belongs there exactly when it can be
    used without a `SimulationGrid` -- and the way that erodes is one convenient
    upward import at a time. Scanned rather than imported, because importing any
    submodule runs the parent `__init__`, which pulls in the grid regardless.

    `utils` is the exception because it is the shared floor: backend resolution,
    host transfer, and the `Array` protocol, none of which know about a grid
    either. It sits above `numerics` only because the rest of the package needs
    it too.
    """
    tree = ast.parse(source_file.read_text())
    # `level` is the number of leading dots: 2 means `from ..X import`, i.e.
    # reaching out of `numerics` into the package proper.
    reached = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level >= 2
    ]
    assert [module for module in reached if module != "utils"] == []
