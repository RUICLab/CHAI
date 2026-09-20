"""The adaptive module and the project config must agree on the error bound.

They did not: adaptive/selector.py said 0.05 while config/config.py said 0.30.
A 0.05 bound excludes sobel-L2 (0.280) entirely, so every "adaptive" study run
under it was really choosing between L0 and L1 -- and the same campaign data
supported opposite conclusions depending on which constant a script happened to
read. This test exists so that cannot drift apart again.
"""
import os
import re


def _declared_in_config():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config", "config.py"), encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r'ERROR_BOUND\s*=\s*_num\("CHAI_ERROR_BOUND",\s*([\d.]+)\)', text)
    assert m, "could not find ERROR_BOUND in config/config.py"
    return float(m.group(1))


def test_adaptive_bound_matches_project_config():
    from adaptive.selector import ERROR_BOUND
    assert ERROR_BOUND == _declared_in_config(), (
        f"adaptive/selector.py says {ERROR_BOUND}, config/config.py says "
        f"{_declared_in_config()}")


def test_bound_admits_the_intended_levels():
    """0.30 must admit L0/L1/L2 on both benchmarks, and exclude the 0.344 rung."""
    from adaptive.power_model import PowerParams
    from adaptive.selector import BENCHMARKS, ERROR_BOUND, build_levels
    for bench in ("sobel", "fft"):
        levels = build_levels(PowerParams(), BENCHMARKS[bench])
        ok = [l.name for l in levels if l.error <= ERROR_BOUND + 1e-12]
        assert ok == ["L0", "L1", "L2"], f"{bench}: admitted {ok}"
    assert 0.344 > ERROR_BOUND, "the 0.344 rung must stay out of specification"
