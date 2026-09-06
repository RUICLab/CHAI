"""Regression tests from real accepted artifacts that passed a weaker checker.

Every code sample here was produced by the model and ACCEPTED by CHAI while
declaring an operator it did not implement. They are kept verbatim so the
checker cannot silently regress to the token-presence rule that let them
through.
"""

from utils.operator_conformance import validate_operator_conformance


# experiment_results/ar_smoke_.../chai_plus10/.../accepted_..._chai.json
# Declared T21. `max_iterations` is read once to compute `limit`, `limit` is
# never used, the loop condition is untouched, and the return is the ordinary
# function return.
AR_SQRT16_FALSE_T21 = """static uint16_t sqrt16(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int max_iterations = 50;
  /*Knob Variables Declaration End*/
  uint16_t hi = 0xffff;
  uint16_t lo = 0;
  uint16_t mid = ((uint32_t)hi + (uint32_t)lo) >> 1;
  uint32_t s = 0;
  int limit = (max_iterations * 100) / 100;
  while (s != x && hi - lo > 1) {
    mid = ((uint32_t)hi + (uint32_t)lo) >> 1;
    s = (uint32_t)mid * (uint32_t)mid;
    if (s < x) lo = mid; else hi = mid;
  }
  return mid;
}"""

# experiment_results/smoke5_.../chai_plus10/.../accepted_..._chai.json
# Declared T21. `early_exit_enabled` is declared and never read at all.
FFT_UNUSED_EARLY_EXIT = """void fft_radix2(int* x, double complex* X, unsigned int size_vec, unsigned int s) {
  /*Knob Variables Declaration Start*/
  int early_exit_enabled = 0;
  int decimation_factor = 0;
  /*Knob Variables Declaration End*/
  if (size_vec == 1) { X[0] = x[0]; return; }
  int limit1 = (size_vec / 2 * decimation_factor) / 100;
  for (int i = 0; i < limit1; i++) { fft_radix2(x, X, size_vec / 2, 2 * s); }
}"""

GENUINE_T21 = """void solve(int n) {
  /*Knob Variables Declaration Start*/
  int max_iterations = 64;
  /*Knob Variables Declaration End*/
  int iters = 0;
  for (int i = 0; i < n; i++) {
    work(i);
    iters++;
    if (iters >= max_iterations) { break; }
  }
}"""


def test_computed_but_unused_budget_is_not_an_early_exit():
    """The exact artifact CHAI shipped while reporting zero violations."""
    error = validate_operator_conformance(AR_SQRT16_FALSE_T21, {21})
    assert error, "a knob that never reaches the loop condition is not T21"


def test_declared_and_never_read_knob_is_not_an_early_exit():
    error = validate_operator_conformance(FFT_UNUSED_EARLY_EXIT, {21})
    assert error


def test_a_knob_guarded_break_is_accepted():
    assert validate_operator_conformance(GENUINE_T21, {21}) == ""


def test_an_ordinary_return_alone_never_satisfies_an_added_operator():
    """The old rule accepted any function containing `return`."""
    trivial = """int f(int n) {
      /*Knob Variables Declaration Start*/
      int early_exit_threshold = 0;
      /*Knob Variables Declaration End*/
      return n;
    }"""
    assert validate_operator_conformance(trivial, {21})
