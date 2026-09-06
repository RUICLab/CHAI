"""Static conformance checks for declared approximation operators.

The LLM must not obtain hardware approval merely by printing an operator ID.
These checks verify distinctive implementation structure before the candidate
can compile or enter Bayesian optimization.

Coverage
--------
Structural checks originally existed only for the ten added operators
(T21-T30), and hardware-semantic checks only for T2 and T6. Everything else --
T1, T3-T5, T7-T20 -- could be declared with no evidence that the transformation
was implemented at all, which is how a candidate whose only real change was a
loop bound came to be labelled "T2, T21". The ``BASE_OPERATOR_CONTRACTS`` table
below closes that gap for the base catalogue.

These are static checks. They prove that the distinctive *structure* of an
operator is present and knob-controlled; they do not prove the data-flow
semantics are correct. They are a floor, not a proof.
"""

import re


def _has(code, pattern):
    return re.search(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL) is not None


def _strip_comments(code):
    """Remove comments so a check cannot be satisfied by prose alone."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", code, flags=re.DOTALL)


def _knob_names(code):
    """Return knob identifiers declared in the knob block."""
    match = re.search(
        r"/\*\s*Knob Variables Declaration Start\s*\*/(.*?)"
        r"/\*\s*Knob Variables Declaration End\s*\*/",
        code,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return set()
    return set(
        re.findall(r"\b([A-Za-z_]\w*)\s*=", match.group(1))
    )


def _knob_tainted_names(code, passes=3):
    """Return the knobs plus every variable derived from a knob.

    Requiring the knob identifier to appear on the same line as the construct
    it controls is too strict for ordinary C: a plan that computes
    ``int limit = (N * perforation_pct) / 100;`` and then loops to ``limit`` is
    a perfectly valid knob-controlled loop. Propagating the taint through
    assignments catches that without accepting an unrelated expression.
    """
    tainted = _knob_names(code)
    if not tainted:
        return tainted
    body = _strip_comments(code)
    assignments = re.findall(
        r"^\s*(?:[A-Za-z_]\w*\s+)*([A-Za-z_]\w*)\s*=\s*([^;]+);",
        body,
        re.MULTILINE,
    )
    for _ in range(passes):
        grew = False
        for target, expression in assignments:
            if target in tainted:
                continue
            if any(
                re.search(rf"\b{re.escape(name)}\b", expression)
                for name in tainted
            ):
                tainted.add(target)
                grew = True
        if not grew:
            break
    return tainted


def _knob_controlled(code, pattern):
    """True when *pattern* appears in an expression involving a knob-derived value."""
    tainted = _knob_tainted_names(code)
    if not tainted:
        return False
    body = _strip_comments(code)
    for line in body.splitlines():
        if re.search(pattern, line, re.IGNORECASE) and any(
            re.search(rf"\b{re.escape(name)}\b", line) for name in tainted
        ):
            return True
    return False


#: Structural requirement for each base-catalogue operator.
#: ``check`` receives the comment-stripped code and the raw code.
BASE_OPERATOR_CONTRACTS = {
    1: (
        "a loop whose bound or stride is controlled by a declared knob",
        lambda body, raw: _has(body, r"\b(for|while)\s*\(")
        and _knob_controlled(raw, r"\b(for|while)\s*\(|[<>]=?|\+="),
    ),
    2: (
        "a knob-controlled change of numeric representation: an integer "
        "shift, mask, or narrowing cast. Multiplying a floating-point value "
        "by a scale knob is value scaling, not precision scaling",
        lambda body, raw: _knob_controlled(
            raw,
            r"(>>|<<|&\s*0[xX]|\(\s*(?:signed\s+|unsigned\s+)?"
            r"(?:char|short|int|long|int8_t|int16_t|int32_t|"
            r"uint8_t|uint16_t|uint32_t)\s*\))",
        ),
    ),
    3: (
        "a cache/table lookup guarded by a hit test before recomputation",
        lambda body, raw: _has(body, r"\b(cache|memo|lookup|lut|_tab)\w*\b")
        and _has(body, r"\bif\s*\("),
    ),
    4: (
        "a knob-controlled branch that skips work",
        lambda body, raw: _has(body, r"\bif\s*\(")
        and _knob_controlled(raw, r"\b(if|continue|goto)\b|[<>=!]="),
    ),
    5: (
        "an approximate container (filter/hash/skip structure) with a knob",
        lambda body, raw: _has(body, r"\b(bloom|filter|hash|skip|sketch)\w*\b")
        and bool(_knob_names(raw)),
    ),
    7: (
        "a knob-controlled early termination of an iterative computation",
        lambda body, raw: _has(body, r"\b(break|return)\b")
        and _knob_controlled(raw, r"\b(break|if)\b|[<>]=?"),
    ),
    8: (
        "a knob-selected simpler computation path",
        lambda body, raw: _has(body, r"\bif\s*\(")
        and _knob_controlled(raw, r"\bif\s*\("),
    ),
    9: (
        "a run-time precision switch driven by a knob",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and _has(body, r"(>>|<<|\(\s*(int|long|float|double)\s*\))"),
    ),
    10: (
        "a knob-controlled quantisation step (shift, mask, or divide-scale)",
        lambda body, raw: _knob_controlled(raw, r"(>>|<<|&|/|\*)"),
    ),
    11: (
        "a knob that trades output fidelity for reduced work",
        lambda body, raw: bool(_knob_names(raw))
        and _has(body, r"\b(if|for|while)\s*\("),
    ),
    12: (
        "a knob-controlled reduction of the data actually processed",
        lambda body, raw: _knob_controlled(raw, r"\[|\+=|[<>]=?"),
    ),
    13: (
        "a stored previous result reused under a knob-controlled condition",
        lambda body, raw: _has(body, r"\b(prev|last|cached|stored)\w*\b")
        and _has(body, r"\bif\s*\("),
    ),
    14: (
        "two computation paths of different fidelity selected by a knob",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and len(re.findall(r"\belse\b", body)) >= 1,
    ),
    16: (
        "a knob-controlled skip of zero or negligible elements",
        lambda body, raw: _has(body, r"\bif\s*\([^)]*(==\s*0|!=\s*0|<\s*\w+)")
        and bool(_knob_names(raw)),
    ),
    19: (
        "a cheaper surrogate replacing an expensive call, under a knob",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and _has(body, r"\b(approx|surrogate|poly|taylor|lookup|lut)\w*\b"),
    ),
    20: (
        "behaviour that adapts to the input under a knob-controlled test",
        lambda body, raw: _knob_controlled(raw, r"\bif\s*\(")
        and _has(body, r"\bif\s*\("),
    ),
}


def nonconforming_base_operator_ids(code, technique_ids):
    """Return base-catalogue IDs whose required structure is absent."""
    body = _strip_comments(code)
    invalid = set()
    for technique_id in sorted(set(technique_ids) & set(BASE_OPERATOR_CONTRACTS)):
        _requirement, check = BASE_OPERATOR_CONTRACTS[technique_id]
        try:
            ok = bool(check(body, code))
        except Exception:
            ok = False
        if not ok:
            invalid.add(technique_id)
    return invalid


def nonconforming_added_operator_ids(code, technique_ids):
    """Return added-operator IDs whose distinctive structure is absent.

    Keeping this separate from the rendered error lets the pipeline remove an
    unsupported *extra claim* while retaining a candidate that also contains a
    real operator.  A candidate whose only claim is unsupported is still
    rejected.
    """
    invalid = set()
    for technique_id in sorted(set(technique_ids) & set(range(21, 31))):
        if technique_id == 21:
            ok = _has(code, r"\b(?:break|return)\b") and _has(
                code, r"\b(?:threshold|max_iterations|iteration_budget|early_exit)\w*\b"
            )
            requirement = "an early break/return controlled by a threshold or iteration-budget knob"
        elif technique_id == 22:
            ok = _has(code, r"\b(?:downsample|spatial|pixel_stride)\w*\b") and _has(
                code, r"\b(?:fill|interpolat|nearest|hold)\w*\b"
            )
            requirement = "a spatial stride/downsample knob plus explicit skipped-output filling"
        elif technique_id == 23:
            ok = _has(code, r"\b(?:decimation|sample_stride|temporal_stride)\w*\b") and _has(
                code, r"\b(?:reconstruct|interpolat|zero_order|hold)\w*\b"
            )
            requirement = "a temporal decimation/stride knob plus reconstruction"
        elif technique_id == 24:
            ok = _has(code, r"\b(?:cabs|hypot|magnitude)\w*\s*\(")
            requirement = "magnitude extraction (for example cabs/hypot) from FFT values"
        elif technique_id == 25:
            ok = _has(code, r"\b(?:lazy|cache|preprocess_budget)\w*\b") and _has(
                code, r"\bif\s*\("
            )
            requirement = "a lazy/cache/budget knob guarding deferred preprocessing"
        elif technique_id == 26:
            ok = _has(code, r">>\s*[A-Za-z_]\w*") and _has(
                code, r"\b(?:ewma|estimate|average|sample)\w*\b"
            )
            requirement = "an EWMA/estimate recurrence using an integer right-shift knob"
        elif technique_id == 27:
            has_lut = _has(
                code,
                r"(?:char|uint8_t|int)\s+[A-Za-z_]\w*\s*\[\s*16\s*\]\s*=\s*\{",
            )
            has_nibble_index = _has(code, r"&\s*0[xX]0?[fF]\b") and (
                _has(code, r"\*\s*4\s*\)") or _has(code, r">>\s*4\b")
            )
            has_stride = _has(code, r"\bnibble_stride\b")
            ok = has_lut and has_nibble_index and has_stride
            requirement = (
                "a local initialized 16-entry LUT, 4-bit (& 0xF) indexing, "
                "and a nibble_stride knob"
            )
        elif technique_id == 28:
            ok = _has(code, r"\b(?:tier|gate|hfe_)\w*\b") and _has(
                code, r"\b(?:fft|spectral|feature)\w*\b"
            )
            requirement = "tier/gate logic controlling expensive spectral features"
        elif technique_id == 29:
            ok = _has(code, r"\bradix_mode\b") and _has(
                code, r"(?:>>\s*[234]\b|&\s*0[xX][fF7])"
            )
            requirement = "a radix_mode knob with power-of-two shift/mask conversion"
        else:  # T30
            ok = _has(code, r"\b(?:segment_len|change_threshold|segment_)\w*\b") and _has(
                code, r"\b(?:fill|hold|interpolat|representative|anchor)\w*\b"
            )
            requirement = "segment/change knobs plus representative-value filling"

        if not ok:
            invalid.add(technique_id)

    return invalid


def validate_operator_conformance(code, technique_ids):
    """Return an actionable error when declared operators are not present."""
    invalid = nonconforming_added_operator_ids(code, technique_ids)
    errors = []

    # Base-catalogue operators used to be accepted on the strength of the label
    # alone. Require their distinctive structure too.
    for technique_id in sorted(nonconforming_base_operator_ids(code, technique_ids)):
        requirement, _check = BASE_OPERATOR_CONTRACTS[technique_id]
        errors.append(f"T{technique_id} requires {requirement}")

    requirements = {
        21: "an early break/return controlled by a threshold or iteration-budget knob",
        22: "a spatial stride/downsample knob plus explicit skipped-output filling",
        23: "a temporal decimation/stride knob plus reconstruction",
        24: "magnitude extraction (for example cabs/hypot) from FFT values",
        25: "a lazy/cache/budget knob guarding deferred preprocessing",
        26: "an EWMA/estimate recurrence using an integer right-shift knob",
        27: "a local initialized 16-entry LUT, 4-bit (& 0xF) indexing, and a nibble_stride knob",
        28: "tier/gate logic controlling expensive spectral features",
        29: "a radix_mode knob with power-of-two shift/mask conversion",
        30: "segment/change knobs plus representative-value filling",
    }
    for technique_id in sorted(invalid):
        errors.append(f"T{technique_id} requires {requirements[technique_id]}")

    if not errors:
        return ""
    return (
        "Operator conformance validation failed: the declared operator ID does "
        "not match the generated C implementation. " + "; ".join(errors)
    )


def validate_hardware_operator_conformance(code, technique_ids, hw_profile):
    """Validate implementation semantics that depend on target capabilities.

    Catalogue feasibility alone is insufficient for T2 on a no-FPU target:
    multiplying a floating-point result by a knob is value scaling, not a
    lower-cost precision implementation.  Require a knob-controlled integer
    or fixed-point path.  T6 likewise needs real probabilistic control rather
    than a label attached to deterministic arithmetic.
    """
    if hw_profile is None:
        return ""

    claimed = set(technique_ids)
    errors = []

    if 2 in claimed and not hw_profile.has_fpu:
        has_precision_mode = _has(
            code,
            r"\b(?:precision_mode|use_integer_math|fixed_point(?:_mode)?|"
            r"precision_bits|fractional_bits)\b",
        )
        has_controlled_path = _has(
            code,
            r"\bif\s*\([^)]*\b(?:precision_mode|use_integer_math|"
            r"fixed_point(?:_mode)?|precision_bits|fractional_bits)\b[^)]*\)",
        )
        has_fixed_arithmetic = _has(
            code,
            r"(?:\b(?:int32_t|uint32_t|long)\b|\(\s*(?:long|int32_t|uint32_t)\s*\)|"
            r"(?:<<|>>)\s*\w+)",
        )
        if not (has_precision_mode and has_controlled_path and has_fixed_arithmetic):
            errors.append(
                "T2 on a no-FPU target requires a knob-controlled integer/fixed-point "
                "arithmetic path; floating-point output scaling is not precision scaling"
            )

    if 6 in claimed:
        has_probability_knob = _has(
            code,
            r"\b(?:probability|sample_rate|random_threshold|lfsr|rng|seed)\w*\b",
        )
        has_probability_logic = _has(
            code,
            r"\b(?:rand|random|lfsr|rng)\w*\s*(?:\(|=)|"
            r"\bif\s*\([^)]*\b(?:probability|sample_rate|random_threshold)\w*\b",
        )
        if not (has_probability_knob and has_probability_logic):
            errors.append(
                "T6 requires an implemented random/pseudorandom decision controlled "
                "by a probability or sampling knob"
            )

    # A no-FPU target pays a software-emulation penalty for every float
    # operation. The approximation prompt itself used to demonstrate loop
    # perforation as `i < (1000 * 0.8)`, which *introduces* floating point on a
    # device that has none. Reject a knob that is declared as a float.
    if not hw_profile.has_fpu:
        knob_block = re.search(
            r"/\*\s*Knob Variables Declaration Start\s*\*/(.*?)"
            r"/\*\s*Knob Variables Declaration End\s*\*/",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if knob_block:
            float_knobs = re.findall(
                r"\b(?:float|double)\s+([A-Za-z_]\w*)", knob_block.group(1)
            )
            if float_knobs:
                errors.append(
                    "knob(s) "
                    + ", ".join(sorted(set(float_knobs)))
                    + " are declared as floating point on a target with no FPU; "
                    "every use costs software-emulated FP cycles. Express the "
                    "knob as an integer (for example a percentage in 0..100 or "
                    "a shift amount) and use integer arithmetic"
                )

    if not errors:
        return ""
    return "Hardware-specific operator conformance failed: " + "; ".join(errors)


def prune_unimplemented_added_operator_claims(function_dict, technique_ids):
    """Drop false added-operator labels when another claimed operator remains.

    This does not invent provenance. It only removes T21-T30 IDs that fail the
    static implementation contract. If removal would leave no claimed
    operator, no mutation occurs and normal validation rejects the candidate.
    """
    claimed = set(technique_ids)
    invalid = nonconforming_added_operator_ids(
        function_dict.get("completeFunction", ""), claimed
    )
    retained = claimed - invalid
    if not invalid or not retained:
        return claimed, []

    function_dict["technique_numbers"] = sorted(retained)
    action = (
        "discarded unimplemented operator claim(s): "
        + ", ".join(f"T{identifier}" for identifier in sorted(invalid))
    )
    return retained, [action]


def repair_operator_candidate(function_dict, technique_ids, hw_profile=None):
    """Install deterministic backends for operators with a canonical C form.

    T27 is a concrete 16-entry population-count transform, not an open-ended
    synthesis task. Keeping its implementation here prevents a small model
    from inventing an undeclared LUT or unsafe recursion while preserving the
    LLM's role in selecting the operator.
    """
    claimed = set(technique_ids)

    # On MSP430-class no-FPU targets, T2 must replace software-emulated
    # floating-point arithmetic with an integer/fixed-point realization. Sobel
    # has a canonical, algebraically equivalent integer form, so use it when a
    # model selects T2 but emits only floating-point value scaling.
    if (
        2 in claimed
        and function_dict.get("functionName") == "sobel_filtering"
        and hw_profile is not None
        and not hw_profile.has_fpu
        and validate_hardware_operator_conformance(
            function_dict.get("completeFunction", ""), {2}, hw_profile
        )
    ):
        function_dict["completeFunction"] = """void sobel_filtering(void)
{
  /*Knob Variables Declaration Start*/
  int precision_mode = 0;
  /*Knob Variables Declaration End*/

  int weight[3][3] = {{ -1,  0,  1 },
                      { -2,  0,  2 },
                      { -1,  0,  1 }};
  int x, y, i, j;

  if (precision_mode == 0) {
    double pixel_value;
    double min_value = DBL_MAX;
    double max_value = -DBL_MAX;

    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        pixel_value = 0.0;
        for (j = -1; j <= 1; j++) {
          for (i = -1; i <= 1; i++) {
            pixel_value += weight[j + 1][i + 1] * image1[y + j][x + i];
          }
        }
        if (pixel_value < min_value) min_value = pixel_value;
        if (pixel_value > max_value) max_value = pixel_value;
      }
    }
    if ((int)(max_value - min_value) == 0) {
      return;
    }

    x_size2 = x_size1;
    y_size2 = y_size1;
    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        pixel_value = 0.0;
        for (j = -1; j <= 1; j++) {
          for (i = -1; i <= 1; i++) {
            pixel_value += weight[j + 1][i + 1] * image1[y + j][x + i];
          }
        }
        pixel_value = MAX_BRIGHTNESS * (pixel_value - min_value) /
                      (max_value - min_value);
        image2[y][x] = (unsigned char)pixel_value;
      }
    }
  } else {
    int pixel_value_int;
    int min_int = 32767;
    int max_int = -32768;

    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        pixel_value_int =
            ((int)image1[y - 1][x + 1] - (int)image1[y - 1][x - 1]) +
            2 * ((int)image1[y][x + 1] - (int)image1[y][x - 1]) +
            ((int)image1[y + 1][x + 1] - (int)image1[y + 1][x - 1]);
        if (pixel_value_int < min_int) min_int = pixel_value_int;
        if (pixel_value_int > max_int) max_int = pixel_value_int;
      }
    }
    if (max_int == min_int) {
      return;
    }

    x_size2 = x_size1;
    y_size2 = y_size1;
    for (y = 1; y < y_size1 - 1; y++) {
      for (x = 1; x < x_size1 - 1; x++) {
        long scaled_value;
        pixel_value_int =
            ((int)image1[y - 1][x + 1] - (int)image1[y - 1][x - 1]) +
            2 * ((int)image1[y][x + 1] - (int)image1[y][x - 1]) +
            ((int)image1[y + 1][x + 1] - (int)image1[y + 1][x - 1]);
        scaled_value =
            (long)MAX_BRIGHTNESS * (long)(pixel_value_int - min_int);
        image2[y][x] =
            (unsigned char)(scaled_value / (long)(max_int - min_int));
      }
    }
  }
}"""
        function_dict["knobVariables"] = "['precision_mode']"
        function_dict["knobRanges"] = "[{'precision_mode': [0, 1]}]"
        function_dict["knobStepSize"] = "[{'precision_mode': 'Integer'}]"
        function_dict["technique_numbers"] = [2]
        return ["installed deterministic no-FPU T2 fixed-point Sobel backend"]

    if 27 not in claimed:
        return []
    if function_dict.get("functionName") != "btbl_bitcnt":
        return []
    if validate_operator_conformance(function_dict.get("completeFunction", ""), {27}) == "":
        return []

    function_dict["completeFunction"] = """int btbl_bitcnt(uint32_t x) {
  /*Knob Variables Declaration Start*/
  int nibble_stride = 1;
  /*Knob Variables Declaration End*/

  static const char nibble_lut[16] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
  };
  int cnt = 0;
  int nibbles_counted = 0;
  for (int i = 0; i < 8; i += nibble_stride) {
    cnt += nibble_lut[(x >> (i * 4)) & 0xF];
    nibbles_counted++;
  }
  if (nibble_stride > 1 && nibbles_counted > 0) {
    cnt = (cnt * 8 + nibbles_counted / 2) / nibbles_counted;
  }
  return cnt;
}"""
    function_dict["knobVariables"] = "['nibble_stride']"
    function_dict["knobRanges"] = "[{'nibble_stride': [1, 4]}]"
    function_dict["knobStepSize"] = "[{'nibble_stride': 'Integer'}]"
    function_dict["technique_numbers"] = [27]
    return ["installed deterministic T27 nibble-lookup backend"]
