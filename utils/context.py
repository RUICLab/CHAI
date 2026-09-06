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
)


def _full_parent_context_enabled():
    return os.getenv("CHAI_FULL_PARENT_CONTEXT", "0").strip().lower() not in (
        "", "0", "false", "no", "off"
    )


#: Maximum characters of a parent's approximated code to carry forward.
PARENT_CODE_BUDGET = int(os.getenv("CHAI_PARENT_CODE_BUDGET", "1200"))


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
