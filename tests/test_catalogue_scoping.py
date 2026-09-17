"""The operator catalogue must not offer categorically inapplicable operators."""

import pytest

from lib.technique_registry import (
    TECHNIQUE_REGISTRY,
    get_applicable_techniques,
)

BENCHMARKS = [
    "ar-iclib", "sobel-iclib", "stringsearch-iclib",
    "fft-iclib", "lqi-iclib", "bc-iclib",
]

#: Operators with no meaning in a single-core, bare-metal C benchmark.
CATEGORICALLY_INAPPLICABLE = {
    15: "relaxed consistency for distributed systems",
    17: "neural-network pruning",
    18: "approximate database joins",
}


@pytest.mark.parametrize("app", BENCHMARKS)
@pytest.mark.parametrize("technique_id", sorted(CATEGORICALLY_INAPPLICABLE))
def test_inapplicable_operators_are_never_offered(app, technique_id):
    """These used to reach every benchmark because T1-T20 had no app tags."""
    assert technique_id not in get_applicable_techniques(app), (
        f"T{technique_id} ({CATEGORICALLY_INAPPLICABLE[technique_id]}) "
        f"must not be offered to {app}"
    )


def test_empty_applicable_apps_is_not_treated_as_universal():
    """`applicable_apps=[]` means 'nothing', not 'everything'."""
    for technique_id in CATEGORICALLY_INAPPLICABLE:
        assert TECHNIQUE_REGISTRY[technique_id].applicable_apps == []

    for app in BENCHMARKS:
        offered = set(get_applicable_techniques(app))
        assert not (offered & set(CATEGORICALLY_INAPPLICABLE))

    # T25 is withdrawn until paired transformations exist; T27 was dominated
    # at every measured stride on its only six-suite target.
    assert TECHNIQUE_REGISTRY[25].applicable_apps == []
    assert TECHNIQUE_REGISTRY[27].applicable_apps == []


@pytest.mark.parametrize("app", BENCHMARKS)
def test_core_operators_remain_available_everywhere(app):
    """Narrowing the catalogue must not strip the operators that do apply."""
    offered = set(get_applicable_techniques(app))
    assert {1, 2, 4, 7, 10, 21}.issubset(offered)


def test_benchmark_specific_operators_stay_scoped_to_their_benchmark():
    assert 27 not in get_applicable_techniques("bc-iclib")
    assert 27 not in get_applicable_techniques("fft-iclib")
    assert 22 in get_applicable_techniques("sobel-iclib")
    assert 22 not in get_applicable_techniques("bc-iclib")
    # T26 replaces an existing EWMA recurrence. lqi-iclib accumulates a
    # delivery ratio in double arithmetic and contains no EWMA, so offering
    # T26 there gave the planner a transformation with no matching structure.
    assert 26 not in get_applicable_techniques("lqi-iclib")
    assert 26 not in get_applicable_techniques("sobel-iclib")
