"""Central accounting for every LLM invocation the pipeline makes.

Reported API-call counts previously covered only the four call sites inside
``main.py``'s per-function loop.  They omitted target selection, the code-base
summary, per-function operator selection, every internal retry in
``approximateFunction``/``convertJson``, and the up-to-ten Makefile-generation
calls per compile test.  On a compile-heavy run that understated real model
usage by roughly an order of magnitude, which matters because model calls are
the cost axis the comparison claims to care about.

Every module that talks to a model records here, tagged by purpose, so the run
report can show both the headline total and where it went.
"""

from collections import Counter
from threading import Lock

_counts = Counter()
_lock = Lock()


def record(stage, count=1):
    """Record *count* model invocations attributed to *stage*."""
    with _lock:
        _counts[stage] += count


def reset():
    """Clear all counters (used at the start of a run)."""
    with _lock:
        _counts.clear()


def snapshot():
    """Return a plain dict of every recorded stage."""
    with _lock:
        return dict(_counts)


def total():
    """Return the total number of recorded model invocations."""
    with _lock:
        return sum(_counts.values())


def format_report():
    """Return a human-readable breakdown, most-used stage first."""
    data = snapshot()
    if not data:
        return "  (no LLM calls recorded)"
    width = max(len(stage) for stage in data)
    lines = []
    for stage, count in sorted(data.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {stage:<{width}}  {count}")
    lines.append(f"    {'TOTAL':<{width}}  {sum(data.values())}")
    return "\n".join(lines)
