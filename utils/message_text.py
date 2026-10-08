"""Normalise LangChain message content to plain text.

Why this exists
---------------
Older providers returned ``message.content`` as a string. Claude models with
extended thinking return a *list of content blocks* instead -- a ``thinking``
block carrying a base64 signature, then the ``text`` block with the actual
answer. Every call site here did ``response.content if hasattr(...) else
str(response)``, so the whole list was stringified.

Two things broke silently as a result:

* Provenance extraction. ``APPLIED_TECHNIQUES:`` has to start a line, and in
  ``[{'type': 'thinking', ...}, {'type': 'text', 'text': "APPLIED_TECHNIQUES..."}]``
  it no longer does. Every candidate was rejected for "missing provenance"
  even though the model had declared its operators correctly.
* Context size. The base64 thinking signatures went into the conversation
  history and were resent on every later call, taking the measured peak prompt
  from 13.4K tokens to 33.7K.

Extracting the text blocks fixes both. Thinking blocks are deliberately
dropped: they are not part of the answer, they must not be re-sent as prose,
and the raw chain of thought is never exposed anyway.
"""


def message_text(message):
    """Return the plain-text content of a model response or raw value."""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Keep text; drop thinking/redacted_thinking/tool blocks.
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        if parts:
            return "\n".join(parts)

    return str(content)
