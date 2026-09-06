"""Context routing for compiler-grounded LLM repair attempts."""


def generation_attempt_context(base_context, attempt):
    """Use full planning context once, then isolate compiler repair attempts."""
    return list(base_context) if attempt == 1 else []


def json_conversion_context(base_context, approximation_conversation, attempt):
    """Prevent JSON conversion from retrieving an older rejected candidate."""
    if attempt == 1:
        return list(base_context) + list(approximation_conversation)
    return list(approximation_conversation)


