import pytest

from utils.error_analyzer import _parse_stringsearch_output


def test_stringsearch_parser_excludes_and_validates_final_match_count():
    assert _parse_stringsearch_output(["1", "0", "1", "2"], "reference") == [
        "1",
        "0",
        "1",
    ]


def test_stringsearch_parser_rejects_non_binary_query_output():
    with pytest.raises(RuntimeError, match="expected '0' or '1'"):
        _parse_stringsearch_output(["1", "garbage", "1"], "candidate")


def test_stringsearch_parser_rejects_incorrect_final_match_count():
    with pytest.raises(RuntimeError, match="decision rows contain 1 matches"):
        _parse_stringsearch_output(["1", "0", "2"], "candidate")


def test_stringsearch_parser_requires_decisions_and_summary():
    with pytest.raises(RuntimeError, match="missing decisions or summary"):
        _parse_stringsearch_output(["0"], "candidate")
