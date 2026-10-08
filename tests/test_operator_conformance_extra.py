"""Regressions from a 7B-vs-32B comparison on sobel_filtering."""

from utils.direct_extraction import extract_c_function
from utils.operator_conformance import validate_operator_conformance

DEAD_KNOB = """void f() {
  /*Knob Variables Declaration Start*/
  int iteration_pct = 100;
  /*Knob Variables Declaration End*/
  double acc;
  int limit = (1000 * iteration_pct) / 100;
  for (int y = 1; y < y_size1 - 1; y++) {
    acc = 0.0;
    for (int j = -1; j <= 1; j++) { acc += w[j] * img[y + j]; }
  }
}"""

LIVE_KNOB = """void f() {
  /*Knob Variables Declaration Start*/
  int iteration_pct = 100;
  /*Knob Variables Declaration End*/
  int limit_y = ((y_size1 - 2) * iteration_pct) / 100;
  for (int y = 1; y < 1 + limit_y; y++) { work(y); }
}"""


def test_knob_computed_then_ignored_is_not_loop_perforation():
    """A knob folded into a local the loop never reads does not implement T1.

    Taint used to spread through `+=` into an unrelated accumulator, so any
    loop mentioning that accumulator satisfied T1 and a dead knob passed.
    """
    assert validate_operator_conformance(DEAD_KNOB, {1})


def test_knob_in_the_loop_header_is_loop_perforation():
    assert validate_operator_conformance(LIVE_KNOB, {1}) == ""


def test_extraction_survives_comments_between_signature_and_brace():
    """`void f( ) /* doc */ {` is ordinary C; requiring `)\\s*{` dropped it."""
    source = """void sobel_filtering( )
     /* Spatial filtering of image data */
     /* Input: image1[y][x] */
{
  int x = 1;
}"""
    extracted = extract_c_function(source, "sobel_filtering")
    assert extracted, "function with comments before the brace must be extracted"
    assert extracted.count("{") == extracted.count("}")
