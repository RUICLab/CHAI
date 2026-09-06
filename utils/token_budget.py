"""Measure the context CHAI actually sends, instead of guessing at it.

``verify_ollama_context`` only checked that the configured ``num_ctx`` was at
least some constant. Nothing ever measured a real prompt, so the choice of a
24K window rested on an unmeasured Modelfile comment ("roughly 15k tokens").
This records the size of every assembled context and writes a summary at the
end of a run, which is the evidence needed before changing the window.

Token counts are estimated from characters when no tokenizer is available. The
estimate is deliberately conservative (3.5 chars/token for code-heavy text).
"""

import json
import os

#: Conservative characters-per-token ratio for mixed C and prose.
CHARS_PER_TOKEN = float(os.getenv("CHAI_CHARS_PER_TOKEN", "3.5"))

_observations = []
_measured = []


def _context_chars(context):
    total = 0
    for entry in context:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            total += len(str(entry[1]))
        else:
            total += len(str(entry))
    return total


def estimate_tokens(chars):
    return int(chars / CHARS_PER_TOKEN)


def record_context(function_name, context, parent_count=0):
    """Record the assembled context size for one function."""
    chars = _context_chars(context)
    _observations.append({
        "function": function_name,
        "messages": len(context),
        "chars": chars,
        "estimated_tokens": estimate_tokens(chars),
        "parents": parent_count,
    })


def reset():
    _observations.clear()
    _measured.clear()


def observations():
    return list(_observations)


def summary():
    """Return aggregate statistics, or ``None`` when nothing was recorded."""
    if not _observations:
        return None
    tokens = [entry["estimated_tokens"] for entry in _observations]
    peak = max(_observations, key=lambda entry: entry["estimated_tokens"])
    return {
        "contexts_measured": len(_observations),
        "chars_per_token_assumed": CHARS_PER_TOKEN,
        "min_estimated_tokens": min(tokens),
        "max_estimated_tokens": max(tokens),
        "mean_estimated_tokens": int(sum(tokens) / len(tokens)),
        "peak_function": peak["function"],
        "peak_parents": peak["parents"],
    }


def write_report(path):
    """Persist per-function measurements and the aggregate summary."""
    data = {
        "estimated_summary": summary(),
        "measured_summary": measured_summary(),
        "observations": observations(),
        "measured_calls": list(_measured),
    }
    if data["estimated_summary"] is None and data["measured_summary"] is None:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def format_summary():
    data = summary()
    if data is None:
        return "  (no contexts measured)"
    return (
        f"    contexts measured : {data['contexts_measured']}\n"
        f"    estimated tokens  : min {data['min_estimated_tokens']}, "
        f"mean {data['mean_estimated_tokens']}, "
        f"max {data['max_estimated_tokens']}\n"
        f"    peak context      : {data['peak_function']} "
        f"({data['peak_parents']} parents)"
    )

# ---------------------------------------------------------------------------
#  Real token counts reported by the provider
# ---------------------------------------------------------------------------
#
# Everything above is an *estimate* from character counts and never sees the
# fully assembled prompt (system messages, few-shot blocks, and the per-step
# template are added after ``manufacturerContext`` returns). It cannot settle
# whether 24K is necessary. Ollama reports ``prompt_eval_count`` -- the actual
# token count of the prompt it evaluated -- and LangChain surfaces it on the
# response. Record those whenever they are available; they are the only numbers
# that should be used to resize ``num_ctx``.


def record_provider_usage(stage, response):
    """Record real prompt/completion token counts from a model response.

    Accepts a LangChain message; silently does nothing when the provider does
    not report usage, so this can be called unconditionally.
    """
    metadata = getattr(response, "response_metadata", None) or {}
    usage = getattr(response, "usage_metadata", None) or {}

    prompt_tokens = (
        usage.get("input_tokens")
        or metadata.get("prompt_eval_count")
        or (metadata.get("token_usage") or {}).get("prompt_tokens")
    )
    completion_tokens = (
        usage.get("output_tokens")
        or metadata.get("eval_count")
        or (metadata.get("token_usage") or {}).get("completion_tokens")
    )
    if not prompt_tokens:
        return None

    entry = {
        "stage": stage,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens or 0),
    }
    _measured.append(entry)
    return entry


def measured_summary():
    """Aggregate real provider-reported token usage, or ``None``."""
    if not _measured:
        return None
    prompts = [entry["prompt_tokens"] for entry in _measured]
    completions = [entry["completion_tokens"] for entry in _measured]
    peak = max(_measured, key=lambda entry: entry["prompt_tokens"])
    by_stage = {}
    for entry in _measured:
        bucket = by_stage.setdefault(entry["stage"], [])
        bucket.append(entry["prompt_tokens"])
    return {
        "source": "provider_reported",
        "calls_measured": len(_measured),
        "max_prompt_tokens": max(prompts),
        "mean_prompt_tokens": int(sum(prompts) / len(prompts)),
        "min_prompt_tokens": min(prompts),
        "max_completion_tokens": max(completions) if completions else 0,
        "peak_stage": peak["stage"],
        "max_prompt_tokens_by_stage": {
            stage: max(values) for stage, values in by_stage.items()
        },
    }


def format_measured_summary():
    data = measured_summary()
    if data is None:
        return (
            "    (provider reported no token usage; only character estimates "
            "are available, which cannot justify resizing num_ctx)"
        )
    lines = [
        f"    provider-reported over {data['calls_measured']} call(s):",
        f"      prompt tokens : min {data['min_prompt_tokens']}, "
        f"mean {data['mean_prompt_tokens']}, max {data['max_prompt_tokens']}",
        f"      peak stage    : {data['peak_stage']}",
        f"      max completion: {data['max_completion_tokens']}",
    ]
    for stage, value in sorted(
        data["max_prompt_tokens_by_stage"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"        {stage:<28} {value}")
    return "\n".join(lines)
