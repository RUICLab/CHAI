"""Ollama configuration checks used before expensive CHAI runs."""

import json
import re
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def parse_parameter(parameters, name):
    """Return an explicit integer Ollama model parameter, if present."""
    match = re.search(rf"(?m)^{re.escape(name)}\s+(\d+)\s*$", parameters or "")
    return int(match.group(1)) if match else None


def parse_num_ctx(parameters):
    """Return an explicit Ollama ``num_ctx`` parameter, if present."""
    return parse_parameter(parameters, "num_ctx")


def ollama_show_url(openai_base_url):
    """Translate an Ollama OpenAI-compatible base URL to its native show API."""
    parsed = urlsplit(openai_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/api/show", "", ""))


def verify_ollama_context(
    base_url,
    model,
    minimum_context,
    timeout=10,
    expected_prediction_limit=None,
):
    """Fail when the selected model would silently truncate CHAI prompts."""
    if minimum_context <= 0:
        return None
    request = Request(
        ollama_show_url(base_url),
        data=json.dumps({"model": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        model_data = json.load(response)
    configured_context = parse_num_ctx(model_data.get("parameters", ""))
    if configured_context is None:
        raise RuntimeError(
            f"Ollama model {model!r} has no explicit num_ctx setting; "
            f"CHAI requires at least {minimum_context}. Create it from the "
            "repository Modelfile before running the campaign."
        )
    if configured_context < minimum_context:
        raise RuntimeError(
            f"Ollama model {model!r} has num_ctx={configured_context}, but "
            f"CHAI requires at least {minimum_context} to avoid prompt truncation."
        )
    if expected_prediction_limit is not None:
        configured_prediction_limit = parse_parameter(
            model_data.get("parameters", ""), "num_predict"
        )
        if configured_prediction_limit != expected_prediction_limit:
            raise RuntimeError(
                f"Ollama model {model!r} must explicitly set "
                f"num_predict={expected_prediction_limit}; found "
                f"{configured_prediction_limit!r}. Without a model-level limit, "
                "Ollama may ignore the OpenAI max_tokens field."
            )
    return configured_context
