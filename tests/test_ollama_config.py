import pytest

from utils.ollama_config import ollama_show_url, parse_num_ctx, parse_parameter


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ("num_ctx                        24576\ntemperature                    0", 24576),
        ("temperature 0", None),
        ("", None),
    ],
)
def test_parse_num_ctx(parameters, expected):
    assert parse_num_ctx(parameters) == expected


def test_ollama_show_url_from_openai_compatibility_url():
    assert (
        ollama_show_url("http://host.docker.internal:11434/v1")
        == "http://host.docker.internal:11434/api/show"
    )


def test_parse_model_level_prediction_limit():
    parameters = "num_ctx 24576\nnum_predict 2048\ntemperature 0"

    assert parse_parameter(parameters, "num_predict") == 2048
