import os

def _flag(name, default):
    """Read an integer switch from the environment, falling back to *default*."""
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _num(name, default):
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


# Prompt-context switches.
#
# Upstream CheckMate (SYSNET-LUMS/CheckMate) ran with ALL of these at 0 and a
# frontier model, and reported good results. This fork turned them on and
# swapped in a 7B local model, which makes the approximation prompt 4,383 chars
# against CheckMate's 1,253 and the planning prompt 3,061 against 489.
#
# That is worth testing rather than assuming: a small model has more
# instructions to violate and degrades over long context, so the extra guidance
# may cost more than it buys. Observed directly -- the 7B reproduced the worked
# example's `(1000 * iteration_pct) / 100` verbatim while ignoring the function
# it was editing.
APPROXIMATION_CONTEXT = _flag("CHAI_APPROXIMATION_CONTEXT", 1)
GIVE_FORMAT_EXAMPLES = _flag("CHAI_GIVE_FORMAT_EXAMPLES", 0)
GIVE_LOOP_PERF_EXMAPLES = _flag("CHAI_GIVE_LOOP_PERF_EXAMPLES", 1)
GIVE_NEW_TECHNIQUE_EXAMPLES = _flag("CHAI_GIVE_NEW_TECHNIQUE_EXAMPLES", 1)
PRINT_LLM_CONVO = _flag("CHAI_PRINT_LLM_CONVO", 0)

# Accuracy bound. CheckMate used 0.50; this fork tightened it to 0.30, so we
# reject configurations CheckMate would have accepted. Per-benchmark bounds in
# utils/error_analyzer.py override this default.
ERROR_BOUND = _num("CHAI_ERROR_BOUND", 0.30)
REVERSED_PDG = 0
TEXT_PLANING = 1
# Supported providers: OpenAI, Anthropic, HuggingFace, and Ollama.
# CHAI_API_NAME in .env overrides the default without requiring a source edit.
API_NAME = os.getenv("CHAI_API_NAME", "OpenAI")
DEBUG = 0
