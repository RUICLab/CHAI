"""Assemble the conversation context handed to the model for one function.

Context budget
--------------
Measured fixed prompt material on this repository, after scoping the
added-operator few-shot examples to what the gatekeeper leaves feasible:

    before scoping   27,341 chars (every app, all ten examples)
    ar-iclib         21,753     sobel-iclib   22,561
    stringsearch     22,809     fft-iclib     20,668
    lqi-iclib        21,533     bc-iclib      20,759

That is roughly 5.9-6.5K tokens. (The Modelfile's old claim of "roughly 15k
tokens" for the prompts was never measured and is about 2x high.)

The part that actually grows is parent history. Every parent function used to
contribute its entire four-message conversation -- purpose, plan, the full
approximated C, and the *same C again* inside the JSON conversion. On a
nine-function application that is eighteen copies of function bodies, which is
what pushes a run toward the 24K window, not the fixed material.

``summarize_parent_contribution`` keeps what a child function actually needs
from a parent -- its purpose and its final approximated signature/knobs -- and
drops the duplicated bodies and the format-repair chatter. Set
``CHAI_FULL_PARENT_CONTEXT=1`` to restore the old unbounded behaviour for an
ablation.
"""

import os
import re

from utils.utils import getNodeParent, Dprint
from utils import token_budget
from config.config import (
    GIVE_FORMAT_EXAMPLES,
    GIVE_LOOP_PERF_EXMAPLES,
    GIVE_NEW_TECHNIQUE_EXAMPLES
)
from config.globals import (
    GLOBAL_CONTEXT,
    LOOP_PERF_EXAMPLES,
    NEW_TECHNIQUE_EXAMPLES,
    APPROXIMATION_FORMAT_EXAMPLES,
    CHAT_HISTORY,
    CODE_BASE_SUMMARY,
    ENTITIES,
)


def _full_parent_context_enabled():
    return os.getenv("CHAI_FULL_PARENT_CONTEXT", "0").strip().lower() not in (
        "", "0", "false", "no", "off"
    )


#: Maximum characters of a parent's approximated code to carry forward.
PARENT_CODE_BUDGET = int(os.getenv("CHAI_PARENT_CODE_BUDGET", "1200"))
CALLSITE_CONTEXT_BUDGET = int(os.getenv("CHAI_CALLSITE_CONTEXT_BUDGET", "3000"))


def summarize_parent_contribution(function_name, messages):
    """Compress one parent's conversation into what a child actually needs.

    A child function needs to know what its parent does and what interface the
    parent now presents. It does not need the parent's rejected candidates,
    format-repair exchanges, or two verbatim copies of the parent's body.
    """
    if not messages:
        return []

    ai_messages = [str(text) for role, text in messages if role == "ai"]
    if not ai_messages:
        return []

    purpose = ai_messages[0].strip()
    if len(purpose) > 600:
        purpose = purpose[:600].rstrip() + " ..."

    # The last AI turn holds the accepted implementation.
    final = ai_messages[-1].strip()
    signature = ""
    knobs = []
    signature_match = re.search(
        rf"[^\n;{{]*\b{re.escape(function_name)}\s*\([^;{{]*\)", final
    )
    if signature_match:
        signature = signature_match.group(0).strip()
    knob_block = re.search(
        r"/\*\s*Knob Variables Declaration Start\s*\*/(.*?)"
        r"/\*\s*Knob Variables Declaration End\s*\*/",
        final,
        re.DOTALL | re.IGNORECASE,
    )
    if knob_block:
        knobs = re.findall(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);", knob_block.group(1))

    summary = [f"Previously processed function `{function_name}`."]
    if purpose:
        summary.append(f"Purpose: {purpose}")
    if signature:
        summary.append(f"Current signature: {signature}")
    if knobs:
        summary.append(
            "Knobs it now exposes: "
            + ", ".join(f"{name} = {value.strip()}" for name, value in knobs)
        )
    else:
        summary.append("It exposes no knobs.")

    return [
        ("human", f"What was done to `{function_name}`?"),
        ("ai", "\n".join(summary)[:PARENT_CODE_BUDGET]),
    ]


def summarize_direct_callsites(function_name, entities, lines_before=12, lines_after=8):
    """Return bounded source snippets for direct callers of *function_name*.

    Reuse opportunities are properties of calls, not just callees. The old
    context showed ``init_search`` in isolation, so the selector confidently
    (and falsely) said it was called once even though ``main`` invokes it in a
    nested loop. These snippets are source evidence shared by every arm.
    """
    snippets = []
    call = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        caller = entity.get("functionName")
        code = str(entity.get("completeFunction", "") or "")
        if not caller or caller == function_name or not code:
            continue
        lines = code.splitlines()
        hit_lines = [
            index for index, line in enumerate(lines) if call.search(line)
        ]
        for index in hit_lines[:2]:
            start = max(0, index - lines_before)
            end = min(len(lines), index + lines_after + 1)
            snippets.append(
                f"Direct caller `{caller}` around line {index + 1}:\n"
                "```c\n" + "\n".join(lines[start:end]) + "\n```"
            )
    if not snippets:
        return []
    text = (
        f"SOURCE CALL-SITE CONTEXT FOR `{function_name}`\n"
        "Use these real call sites when judging reuse, loop invariance, and "
        "argument stability. Do not assume a call count absent from source.\n\n"
        + "\n\n".join(snippets)
    )
    return [("human", text[:CALLSITE_CONTEXT_BUDGET])]


def _balanced_end(code, opening, left, right):
    depth = 0
    for position in range(opening, len(code)):
        if code[position] == left:
            depth += 1
        elif code[position] == right:
            depth -= 1
            if depth == 0:
                return position
    return len(code) - 1


def _for_regions(code):
    """Return ``(body_start, body_end, induction_variable)`` regions."""
    regions = []
    for match in re.finditer(r"\bfor\s*\(", code):
        opening = code.find("(", match.start())
        close = _balanced_end(code, opening, "(", ")")
        header = code[opening + 1:close]
        induction = re.search(
            r"(?:^|;)\s*(?:[A-Za-z_]\w*[\s*]+)*([A-Za-z_]\w*)\s*=",
            header,
        )
        variable = induction.group(1) if induction else None
        body_start = close + 1
        while body_start < len(code) and code[body_start].isspace():
            body_start += 1
        if body_start < len(code) and code[body_start] == "{":
            body_end = _balanced_end(code, body_start, "{", "}")
        else:
            semicolon = code.find(";", body_start)
            body_end = len(code) if semicolon == -1 else semicolon
        regions.append((body_start, body_end, variable))
    return regions


def loop_invariant_callsite_evidence(function_name, entities):
    """Describe direct calls whose arguments are invariant in an enclosing loop.

    This is a candidate detector, not a semantic proof. Identity and behavioral
    gates still decide whether a generated T3 implementation is safe.
    """
    callee = next(
        (
            str(entity.get("completeFunction", "") or "")
            for entity in entities or []
            if isinstance(entity, dict) and entity.get("functionName") == function_name
        ),
        "",
    )
    callee_body = callee[callee.find("{") + 1:] if "{" in callee else ""
    if not re.search(r"\b(?:for|while)\s*\(|\b[A-Za-z_]\w*\s*\(", callee_body):
        return []

    evidence = []
    call_pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    for entity in entities or []:
        if not isinstance(entity, dict) or entity.get("functionName") == function_name:
            continue
        caller = entity.get("functionName")
        code = str(entity.get("completeFunction", "") or "")
        regions = _for_regions(code)
        for call in call_pattern.finditer(code):
            opening = code.find("(", call.start())
            close = _balanced_end(code, opening, "(", ")")
            arguments = code[opening + 1:close]
            enclosing = [
                region for region in regions if region[0] <= call.start() <= region[1]
            ]
            # Smallest body is the innermost loop.
            enclosing.sort(key=lambda region: region[1] - region[0])
            # Only the innermost loop establishes consecutive equal-key calls.
            # Invariance merely in an outer loop is insufficient: an inner
            # loop may change the key or global state between two calls.
            if enclosing:
                _start, _end, variable = enclosing[0]
                if variable and not re.search(rf"\b{re.escape(variable)}\b", arguments):
                    evidence.append(
                        f"{caller} calls {function_name}({arguments.strip()}) inside "
                        f"its innermost loop over {variable}, but the arguments do not depend on {variable}"
                    )
    return evidence


def enforce_callsite_operator_requirements(candidate_ids, function_name, entities):
    """Remove call-site-only operators when their required evidence is absent.

    The selector prompt tells the model that T3 cannot be inferred from a
    callee body alone, but a model still selected it for ``strsearch`` and the
    generated implementation duplicated the full search in both branches.
    Candidate narrowing is a control-plane decision, so enforce that rule here
    before planning rather than hoping the model follows prose.
    """
    selected = set(candidate_ids or [])
    t3_evidence = loop_invariant_callsite_evidence(function_name, entities)
    removed = set()
    if 3 in selected and not t3_evidence:
        selected.remove(3)
        removed.add(3)
    return selected, removed, t3_evidence


def manufacturerContext(PDG, this_function): #Cat 6

    global APPROXIMATION_FORMAT_EXAMPLES
    global CHAT_HISTORY
    global GIVE_FORMAT_EXAMPLES

    parent_entities = getNodeParent(this_function, PDG)

    parent_context = getConversationHistory(parent_entities, CHAT_HISTORY)

    this_context = GLOBAL_CONTEXT.copy()

    if GIVE_LOOP_PERF_EXMAPLES:
        this_context += LOOP_PERF_EXAMPLES
    if GIVE_NEW_TECHNIQUE_EXAMPLES:
        this_context += NEW_TECHNIQUE_EXAMPLES
    if GIVE_FORMAT_EXAMPLES:
        this_context += APPROXIMATION_FORMAT_EXAMPLES
    if CODE_BASE_SUMMARY != []:
        this_context += CODE_BASE_SUMMARY

    this_context += summarize_direct_callsites(this_function, ENTITIES)

    this_context += parent_context

    token_budget.record_context(this_function, this_context, len(parent_entities))
    return this_context


def getConversationHistory(function_list, chat_history): #Cat 6
    manufactured_history = []
    full = _full_parent_context_enabled()
    for function in function_list:
        messages = chat_history.get(function)
        if not messages:
            Dprint(
                "A converstation that does not exist was attempted to be added: ",
                function,
            )
            continue
        if full:
            manufactured_history += messages
        else:
            manufactured_history += summarize_parent_contribution(function, messages)

    return manufactured_history
