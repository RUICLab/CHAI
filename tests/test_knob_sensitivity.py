"""Behavioural verification replaces a class of static-checker gaps.

Static conformance asks whether code *looks* like an operator. These tests pin
the property actually claimed: changing the knob changes what the program does.
"""

import os

import pytest

from utils.knob_sensitivity import (
    SensitivityUnavailable,
    _declared_value,
    _probe_value,
    _set_knob_in_source,
    requires_output_sensitivity,
    verify_knobs_have_effect,
)

INERT = """#include <stdio.h>
int main(void) {
  /*Knob Variables Declaration Start*/
  int max_iterations = 50;
  /*Knob Variables Declaration End*/
  int limit = (max_iterations * 100) / 100;
  int s = 0;
  for (int i = 0; i < 8; i++) s += i;
  printf("%d\\n", s);
  return 0;
}
"""

EFFECTIVE = """#include <stdio.h>
int main(void) {
  /*Knob Variables Declaration Start*/
  int iteration_pct = 100;
  /*Knob Variables Declaration End*/
  int limit = (8 * iteration_pct) / 100;
  int s = 0;
  for (int i = 0; i < limit; i++) s += i;
  printf("%d\\n", s);
  return 0;
}
"""


def test_probe_value_picks_the_end_furthest_from_identity():
    assert _probe_value([0, 100], 100) == 0
    assert _probe_value([0, 100], 0) == 100
    assert _probe_value([50, 100], 100) == 50


def test_exact_t3_does_not_require_output_change_but_mixed_plans_do():
    assert not requires_output_sensitivity({3})
    assert requires_output_sensitivity({1})
    assert requires_output_sensitivity({3, 21})
    assert requires_output_sensitivity(set())


def test_declared_value_reads_the_knob_block(tmp_path):
    (tmp_path / "main.c").write_text(INERT)
    assert _declared_value(str(tmp_path), "max_iterations") == 50


def test_set_knob_rewrites_only_the_declaration(tmp_path):
    (tmp_path / "main.c").write_text(EFFECTIVE)
    assert _set_knob_in_source(str(tmp_path), "iteration_pct", 25)
    source = (tmp_path / "main.c").read_text()
    assert "int iteration_pct = 25;" in source
    assert "(8 * iteration_pct) / 100" in source, "uses must not be rewritten"


def test_a_knob_that_is_computed_but_never_used_is_inert(tmp_path):
    """The exact shape of the accepted AR sqrt16 artifact."""
    (tmp_path / "main.c").write_text(INERT)
    assert _declared_value(str(tmp_path), "max_iterations") == 50
    # Driving the knob to the other end must leave the program text that
    # actually executes unchanged, which is what makes it inert.
    assert _set_knob_in_source(str(tmp_path), "max_iterations", 4)
    source = (tmp_path / "main.c").read_text()
    assert "for (int i = 0; i < 8; i++)" in source


def test_missing_candidate_directory_is_unavailable_not_a_pass():
    """Absence of evidence must never be reported as a clean result."""
    with pytest.raises(SensitivityUnavailable):
        verify_knobs_have_effect("fft-iclib", "/nonexistent/dir", {})
