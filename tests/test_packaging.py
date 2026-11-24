"""Checks on the dependencies declared in setup.py (no installation needed)."""
import ast
import os
import re

SETUP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py")


def _requirements():
    tree = ast.parse(open(SETUP).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "requirements" for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError("no requirements list in setup.py")


def _lower_bound(name):
    for req in _requirements():
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", req)
        if m.group(1).lower() == name.lower():
            bound = re.search(r">=\s*([0-9][0-9.]*)", m.group(2))
            return tuple(int(x) for x in bound.group(1).split(".")) if bound else ()
    return None


def test_joblib_is_a_direct_dependency():
    # joblib is imported directly (Parallel, delayed); NumPy 2 support starts in joblib 1.4.0
    bound = _lower_bound("joblib")
    assert bound is not None, "joblib is imported by SigProfilerAssignment but not declared"
    assert bound >= (1, 4, 0)


def test_pandas_lower_bound_supports_numpy_2():
    # pandas 2.2.2 is the first release built against NumPy 2.0
    assert _lower_bound("numpy") >= (2, 0, 0)
    assert _lower_bound("pandas") >= (2, 2, 2)


def test_direct_imports_are_declared():
    # matplotlib (plots) and PIL (decomposition plots) are imported directly
    assert _lower_bound("matplotlib") >= (3, 8, 4)  # first release with wheels built against NumPy 2
    assert _lower_bound("pillow") >= (10, 0, 0)  # as required by sigProfilerPlotting


def test_lower_bounds_install_and_support_all_contexts():
    # sigProfilerPlotting < 1.5.0 does not know the SBS384/SBS4608 contexts;
    # reportlab < 3.5.55 has no Python 3.9 wheels (3.6.0 is the verified minimum)
    assert _lower_bound("sigProfilerPlotting") >= (1, 5, 0)
    assert _lower_bound("reportlab") >= (3, 6, 0)
