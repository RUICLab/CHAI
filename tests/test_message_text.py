"""Claude models with thinking return content blocks, not a string."""

from utils.message_text import message_text


class _Msg:
    def __init__(self, content):
        self.content = content


def test_plain_string_is_unchanged():
    assert message_text(_Msg("APPLIED_TECHNIQUES: T1")) == "APPLIED_TECHNIQUES: T1"


def test_text_blocks_are_extracted_and_thinking_dropped():
    """Stringifying the block list buried the declaration mid-line.

    APPLIED_TECHNIQUES must start a line for provenance extraction, and the
    base64 thinking signature must never reach the conversation history.
    """
    msg = _Msg([
        {"type": "thinking", "thinking": "", "signature": "EqgdCpABCBEYAipA/zbHAPbC"},
        {"type": "text", "text": "APPLIED_TECHNIQUES: T1\n\nvoid f(void) {}"},
    ])
    out = message_text(msg)
    assert out.startswith("APPLIED_TECHNIQUES: T1")
    assert "EqgdCpABCBEYAipA" not in out


def test_provenance_extraction_works_on_normalised_content():
    from lib.hw_validator import extract_technique_ids_from_convo
    blocks = [
        {"type": "thinking", "signature": "abc123"},
        {"type": "text", "text": "APPLIED_TECHNIQUES: T1, T21\ncode here"},
    ]
    assert extract_technique_ids_from_convo([("ai", blocks)]) == set()
    assert extract_technique_ids_from_convo([("ai", message_text(_Msg(blocks)))]) == {1, 21}
