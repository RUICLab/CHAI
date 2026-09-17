from utils.operator_conformance import (
    nonconforming_operator_ids,
    nonconforming_added_operator_ids,
    prune_unimplemented_added_operator_claims,
    prune_unimplemented_operator_claims,
    repair_operator_candidate,
    validate_hardware_operator_conformance,
    validate_operator_conformance,
)
from lib.hardware_profile import HardwareProfile


# The knob-declaration block is mandatory in the generation contract, and
# normalize_knob_metadata depends on it downstream, so conformance requires it
# too: an added operator with no declared, used knob cannot be substantiated.
VALID_T27 = """
int bitcount(unsigned long x) {
  /*Knob Variables Declaration Start*/
  int nibble_stride = 1;
  /*Knob Variables Declaration End*/
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


def test_t3_recognises_keyed_reuse_by_dataflow_not_variable_names():
    keyed_reuse = """
    static const char *remembered_pattern;
    static int jump_values[256];
    void prepare(const char *incoming) {
      /*Knob Variables Declaration Start*/
      int enable_reuse = 1;
      /*Knob Variables Declaration End*/
      if (enable_reuse && remembered_pattern == incoming) return;
      int width = strlen(incoming);
      for (int i = 0; i < 256; ++i) jump_values[i] = width;
      remembered_pattern = (const char *)incoming;
    }
    """

    assert validate_operator_conformance(keyed_reuse, {3}) == ""

    # Alpha-renaming every semantic object must not change the verdict.
    renamed = (keyed_reuse
        .replace("remembered_pattern", "xq")
        .replace("incoming", "zz")
        .replace("jump_values", "a7")
        .replace("enable_reuse", "k9"))
    assert validate_operator_conformance(renamed, {3}) == ""


def test_t3_rejects_cache_words_without_a_real_hit_and_miss_path():
    keyword_only = """
    int f(int input) {
      /*Knob Variables Declaration Start*/
      int use_cache = 1;
      /*Knob Variables Declaration End*/
      int cache = input;
      if (use_cache) return cache;
      return input;
    }
    """

    assert "T3 requires" in validate_operator_conformance(keyword_only, {3})


def test_t3_rejects_a_hit_branch_that_recomputes_before_returning():
    """Duplicating the full computation in both branches is not exact reuse."""
    duplicated_search = """
    int search(const char *input) {
      /*Knob Variables Declaration Start*/
      int reuse_enabled = 1;
      /*Knob Variables Declaration End*/
      static const char *saved_key;
      static int saved_len;
      if (reuse_enabled && saved_key == input && saved_len == width) {
        for (int i = 0; i < width; ++i) inspect(input[i]);
        return 0;
      } else {
        saved_key = input;
        saved_len = width;
        for (int i = 0; i < width; ++i) inspect(input[i]);
        return 0;
      }
    }
    """

    assert "T3 requires" in validate_operator_conformance(duplicated_search, {3})


def test_t3_and_t13_have_distinct_structural_contracts():
    previous_result = """
    int smooth(int input) {
      /*Knob Variables Declaration Start*/
      int refresh_every = 2;
      /*Knob Variables Declaration End*/
      static int tick = 0;
      static int retained = 0;
      if ((tick++ % refresh_every) == 0) {
        retained = expensive(input);
      }
      emit(retained);
      return retained;
    }
    """

    assert validate_operator_conformance(previous_result, {13}) == ""
    assert "T3 requires" in validate_operator_conformance(previous_result, {3})

    keyed_reuse = """
    static const void *a;
    void f(const void *b) {
      /*Knob Variables Declaration Start*/
      int k = 1;
      /*Knob Variables Declaration End*/
      if (k && a == b) return;
      recompute(b);
      a = b;
    }
    """
    assert validate_operator_conformance(keyed_reuse, {3}) == ""
    assert "T13 requires" in validate_operator_conformance(keyed_reuse, {13})


def test_t5_and_t19_do_not_depend_on_magic_identifier_names():
    compact_membership = """
    int maybe_present(unsigned value) {
      /*Knob Variables Declaration Start*/
      unsigned k1 = 63;
      /*Knob Variables Declaration End*/
      static unsigned char q7[8];
      unsigned z9 = value % k1;
      return q7[z9 >> 3] & (1u << (z9 & 7));
    }
    """
    assert validate_operator_conformance(compact_membership, {5}) == ""

    surrogate = """
    int rootish(unsigned value) {
      /*Knob Variables Declaration Start*/
      int k2 = 0;
      /*Knob Variables Declaration End*/
      if (k2) {
        return (value + 1) >> 1;
      } else {
        return expensive_root(value);
      }
    }
    """
    assert validate_operator_conformance(surrogate, {19}) == ""

    labels_only = """
    int f(int x) {
      /*Knob Variables Declaration Start*/
      int surrogate_mode = 1;
      /*Knob Variables Declaration End*/
      if (surrogate_mode) return x;
      return x;
    }
    """
    assert "T19 requires" in validate_operator_conformance(labels_only, {19})


def test_t6_probability_check_is_alpha_rename_invariant():
    code = """
    void f(void) {
      /*Knob Variables Declaration Start*/
      unsigned k3 = 2;
      /*Knob Variables Declaration End*/
      unsigned q8 = source_state;
      q8 = (q8 >> 1) ^ (-(q8 & 1u) & 0xB400u);
      if ((q8 % k3) == 0) output = work();
    }
    """

    assert validate_hardware_operator_conformance(code, {6}, _msp430_profile()) == ""


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


def test_loop_conformance_handles_casts_and_unbraced_bodies():
    code = """
    void f(unsigned n) {
      /*Knob Variables Declaration Start*/
      int iteration_pct = 100;
      /*Knob Variables Declaration End*/
      int limit = (n * iteration_pct) / 100;
      for (unsigned i = 0; i < (unsigned)limit; ++i) work(i);
    }
    """

    assert validate_operator_conformance(code, {1}) == ""


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


def test_false_base_claim_is_pruned_when_a_real_operator_remains():
    candidate = {
        "completeFunction": """
        void f(int *data, int n) {
          /*Knob Variables Declaration Start*/
          int iteration_pct = 100;
          /*Knob Variables Declaration End*/
          int limit = (n * iteration_pct) / 100;
          for (int i = 0; i < limit; ++i) data[i] += 1;
        }
        """,
        "technique_numbers": [1, 13],
    }

    retained, actions = prune_unimplemented_operator_claims(candidate, {1, 13})

    assert retained == {1}
    assert candidate["technique_numbers"] == [1]
    assert actions == ["discarded unimplemented operator claim(s): T13"]


def test_all_false_claims_are_left_for_normal_rejection():
    candidate = {
        "completeFunction": "int f(void) { return 0; }",
        "technique_numbers": [13, 21],
    }

    retained, actions = prune_unimplemented_operator_claims(candidate, {13, 21})

    assert retained == {13, 21}
    assert actions == []


def test_raw_claim_measurement_covers_base_and_added_catalogues():
    code = "int f(void) { return 0; }"

    assert nonconforming_operator_ids(code, {13, 21}) == {13, 21}


def _msp430_profile():
    return HardwareProfile("MSP430", 10, False, False, 5, 16)


def test_no_fpu_t2_does_not_require_a_magic_knob_name():
    code = """
    void quantize(unsigned *output, unsigned value) {
      /*Knob Variables Declaration Start*/
      int quant_shift = 0;
      /*Knob Variables Declaration End*/
      *output = value >> quant_shift;
    }
    """

    assert validate_operator_conformance(code, {2}) == ""
    assert validate_hardware_operator_conformance(
        code, {2}, _msp430_profile()
    ) == ""


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
      /*Knob Variables Declaration Start*/
      int sample_rate = 2;
      /*Knob Variables Declaration End*/
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


def test_an_added_operator_without_a_declared_knob_is_rejected():
    """An operator with nothing to tune cannot be substantiated.

    The generation contract mandates the knob-declaration block, and the
    optimizer needs it to build a search space, so its absence is a rejection
    rather than a pass.
    """
    no_knob_block = """
    int bitcount(unsigned long x) {
      static const char nibble_lut[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
      int cnt = 0;
      for (int i = 0; i < 8; i++) cnt += nibble_lut[(x >> (i * 4)) & 0xF];
      return cnt;
    }
    """

    assert validate_operator_conformance(no_knob_block, {27})
