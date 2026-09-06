from utils.operator_conformance import (
    nonconforming_added_operator_ids,
    prune_unimplemented_added_operator_claims,
    repair_operator_candidate,
    validate_hardware_operator_conformance,
    validate_operator_conformance,
)
from lib.hardware_profile import HardwareProfile


VALID_T27 = """
int bitcount(unsigned long x) {
  int nibble_stride = 1;
  static const char nibble_lut[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
  int cnt = 0;
  for (int i = 0; i < 8; i += nibble_stride)
    cnt += nibble_lut[(x >> (i * 4)) & 0xF];
  return cnt;
}
"""


def test_t27_requires_real_nibble_lookup_structure():
    mislabeled = """
    int bitcount(unsigned x) {
      int stride = 1;
      return bits[x & 0xFF] + bitcount(x >> 8);
    }
    """

    error = validate_operator_conformance(mislabeled, {27})

    assert "T27 requires" in error
    assert validate_operator_conformance(VALID_T27, {27}) == ""


def test_original_catalog_operators_are_not_judged_by_added_operator_rules():
    """T1/T2 must be checked against their own contract, not T21-T30's."""
    error = validate_operator_conformance("int f(void) { return 0; }", {1, 2})

    # The complaint must be about T1/T2's own requirements ...
    assert "T1 requires" in error and "T2 requires" in error
    # ... and must never mention an added-operator requirement.
    for added in range(21, 31):
        assert f"T{added} requires" not in error


def test_base_operators_require_their_declared_structure():
    """A bare relabel used to be accepted for the whole T1-T20 catalogue."""
    assert "T1 requires" in validate_operator_conformance(
        "int f(void) { return 0; }", {1}
    )

    perforated = """
    void f(int *d, int n) {
      /*Knob Variables Declaration Start*/
      int iteration_pct = 100;
      /*Knob Variables Declaration End*/
      int limit = (n * iteration_pct) / 100;
      for (int i = 0; i < limit; i++) { d[i] = d[i] + 1; }
    }
    """
    assert validate_operator_conformance(perforated, {1}) == ""


def test_t2_rejects_float_scaling_and_accepts_fixed_point():
    """Multiplying a float result by a knob is value scaling, not precision."""
    float_scaling = """
    void f(void) {
      /*Knob Variables Declaration Start*/
      double scale = 1.0;
      /*Knob Variables Declaration End*/
      double x = compute();
      x = x * scale;
      store(x);
    }
    """
    assert "T2 requires" in validate_operator_conformance(float_scaling, {2})

    fixed_point = """
    void f(void) {
      /*Knob Variables Declaration Start*/
      int precision_mode = 0;
      int frac_bits = 8;
      /*Knob Variables Declaration End*/
      if (precision_mode) {
        int32_t acc = accumulate_i();
        acc = (acc * 3) >> frac_bits;
        store_i(acc);
      } else {
        store(compute());
      }
    }
    """
    assert validate_operator_conformance(fixed_point, {2}) == ""


def test_knob_taint_follows_one_level_of_indirection():
    """A loop bound computed from a knob is still knob-controlled."""
    indirect = """
    void f(int n) {
      /*Knob Variables Declaration Start*/
      int keep_pct = 100;
      /*Knob Variables Declaration End*/
      int limit = (n * keep_pct) / 100;
      for (int i = 0; i < limit; i++) { work(i); }
    }
    """
    assert validate_operator_conformance(indirect, {1}) == ""

    unrelated = """
    void f(int n) {
      /*Knob Variables Declaration Start*/
      int enable = 0;
      /*Knob Variables Declaration End*/
      for (int i = 0; i < n; i++) { work(i); }
    }
    """
    assert "T1 requires" in validate_operator_conformance(unrelated, {1})


def test_no_fpu_target_rejects_floating_point_knobs():
    """The prompt used to teach `i < (1000 * 0.8)` on a device with no FPU."""
    profile = HardwareProfile(
        name="MSP430", ram_kb=10, has_fpu=False, has_simd=False,
        energy_budget_mj=5, clock_mhz=16,
    )
    float_knob = """
    void f(int n) {
      /*Knob Variables Declaration Start*/
      double perforation = 1.0;
      /*Knob Variables Declaration End*/
      int limit = (int)(n * perforation);
      for (int i = 0; i < limit; i++) { work(i); }
    }
    """
    error = validate_hardware_operator_conformance(float_knob, {1}, profile)
    assert "no FPU" in error and "perforation" in error


def test_t27_backend_repairs_mislabeled_bitcount_candidate():
    candidate = {
        "functionName": "btbl_bitcnt",
        "completeFunction": "int btbl_bitcnt(uint32_t x) { return bits[x & 0xFF]; }",
        "knobVariables": "[]",
        "knobRanges": "[]",
        "knobStepSize": "[]",
    }

    actions = repair_operator_candidate(candidate, {27})

    assert actions == ["installed deterministic T27 nibble-lookup backend"]
    assert validate_operator_conformance(candidate["completeFunction"], {27}) == ""
    assert candidate["knobVariables"] == "['nibble_stride']"


def test_false_extra_added_operator_claim_is_removed_not_accepted():
    candidate = {
        "completeFunction": (
            "double work(double x) { "
            "double precision_scale = 0.9; return x * precision_scale; }"
        ),
        "technique_numbers": [2, 21],
    }

    retained, actions = prune_unimplemented_added_operator_claims(
        candidate, {2, 21}
    )

    assert nonconforming_added_operator_ids(candidate["completeFunction"], {21}) == {21}
    assert retained == {2}
    assert candidate["technique_numbers"] == [2]
    assert actions == ["discarded unimplemented operator claim(s): T21"]


def test_only_false_operator_claim_is_left_for_rejection():
    candidate = {
        "completeFunction": "int work(void) { return 0; }",
        "technique_numbers": [21],
    }

    retained, actions = prune_unimplemented_added_operator_claims(candidate, {21})

    assert retained == {21}
    assert actions == []
    assert candidate["technique_numbers"] == [21]


def _msp430_profile():
    return HardwareProfile("MSP430", 10, False, False, 5, 16)


def test_no_fpu_t2_rejects_floating_output_scaling():
    code = """
    void sobel(void) {
      double precision_scale = 0.8;
      double pixel_value = convolution();
      output = pixel_value * precision_scale;
    }
    """

    error = validate_hardware_operator_conformance(code, {2}, _msp430_profile())

    assert "integer/fixed-point" in error


def test_no_fpu_t2_accepts_knob_controlled_fixed_point_path():
    code = """
    void sobel(void) {
      int precision_mode = 0;
      if (precision_mode == 0) {
        original_double_path();
      } else {
        long scaled_value = (long)pixel * 255L;
        output = scaled_value / range;
      }
    }
    """

    assert validate_hardware_operator_conformance(
        code, {2}, _msp430_profile()
    ) == ""


def test_t6_requires_real_probability_logic():
    mislabeled = "void f(void) { output = input * 0.8; }"
    implemented = """
    void f(void) {
      int sample_rate = 2;
      unsigned lfsr = seed;
      lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
      if ((lfsr % sample_rate) == 0) output = work();
    }
    """

    assert "T6 requires" in validate_hardware_operator_conformance(
        mislabeled, {6}, _msp430_profile()
    )
    assert validate_hardware_operator_conformance(
        implemented, {6}, _msp430_profile()
    ) == ""


def test_no_fpu_sobel_t2_backend_repairs_invalid_model_realization():
    candidate = {
        "functionName": "sobel_filtering",
        "completeFunction": (
            "void sobel_filtering(void) { double precision_scale = 0.8; "
            "image2[0][0] = image1[0][0] * precision_scale; }"
        ),
        "knobVariables": "['precision_scale']",
        "knobRanges": "[{'precision_scale': [0.5, 1.0]}]",
        "knobStepSize": "[{'precision_scale': 'Real'}]",
        "technique_numbers": [2, 6],
    }

    actions = repair_operator_candidate(candidate, {2, 6}, _msp430_profile())

    assert actions == ["installed deterministic no-FPU T2 fixed-point Sobel backend"]
    assert candidate["technique_numbers"] == [2]
    assert candidate["knobVariables"] == "['precision_mode']"
    assert validate_hardware_operator_conformance(
        candidate["completeFunction"], {2}, _msp430_profile()
    ) == ""


def test_sobel_t2_backend_is_not_used_without_chai_hardware_control():
    candidate = {
        "functionName": "sobel_filtering",
        "completeFunction": "void sobel_filtering(void) { return; }",
    }

    assert repair_operator_candidate(candidate, {2}, None) == []
    assert candidate["completeFunction"] == "void sobel_filtering(void) { return; }"
