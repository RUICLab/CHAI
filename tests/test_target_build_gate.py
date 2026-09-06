"""The acceptance gate must use the target toolchain, not host gcc."""

import utils.target_build as target_build
from utils.target_build import (
    MSP430_REGIONS,
    TargetBuildResult,
    _memory_usage,
    _parse_size_output,
    validate_target_build,
)

SIZE_OUTPUT = """
section              size         addr
.text                8192        16384
.rodata              1024        24576
.data                 256         7168
.bss                  512         7424
Total               10000
"""


def test_size_output_is_parsed_into_sections():
    sizes = _parse_size_output(SIZE_OUTPUT)

    assert sizes[".text"] == 8192
    assert sizes[".bss"] == 512


def test_sections_map_onto_device_regions():
    usage = _memory_usage(_parse_size_output(SIZE_OUTPUT))

    assert usage["RAM"] == 512 + 256
    assert usage["FRAM"] == 8192 + 1024 + 256


def test_device_regions_match_the_linker_script():
    """msp430fr5994.ld declares RAM 0x1000 and the primary FRAM 0xBF80."""
    assert MSP430_REGIONS["RAM"] == 0x1000
    assert MSP430_REGIONS["FRAM"] == 0xBF80


def test_a_missing_toolchain_reports_skipped_not_passed(monkeypatch):
    """A run without the cross compiler must never look like a validated run."""
    monkeypatch.setattr(target_build, "toolchain_available", lambda: False)

    result = validate_target_build("ar-iclib", "compilation_testing/ApxFiles_f")

    assert result.status == "skipped"
    assert "did not run" in result.message
    assert result.ok  # skipped does not block, but it is recorded as skipped


def test_result_states_are_distinguishable():
    assert TargetBuildResult("passed").ok
    assert TargetBuildResult("skipped").ok
    assert not TargetBuildResult("failed", "region FRAM overflowed").ok


def test_overflow_pattern_matches_real_linker_output():
    text = (
        "/usr/bin/msp430-elf-ld: region `FRAM' overflowed by 2048 bytes\n"
        "collect2: error: ld returned 1 exit status"
    )
    match = target_build._OVERFLOW_PATTERN.search(text)

    assert match is not None
    assert match.group(1) == "FRAM"
    assert match.group(2) == "2048"
