import os
import subprocess
import re
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from utils.utils import (
    formatMessageForHistory
)
from lib.lsp import (
    lsp_extractor
)

from config.globals import (
    GLOBAL_CONTEXT, 
    TARGET_FILES, 
    PURPOSE_IDENTIFICATION_SCRIPT,
    ANNOTATION_SCRIPT, 
    APPROXIMATION_SCRIPT, 
    PLANNING_SCRIPT,
    LOOP_PERF_EXAMPLES, 
    NEW_TECHNIQUE_EXAMPLES,
    APPROXIMATION_FORMAT_EXAMPLES,
    TARGET_FUNCTIONS_SCRIPT,
    CODE_BASE_SUMMARY,
)
from config.config import APPROXIMATION_CONTEXT
# Cat 1


def _without_auto_generated_cards(text):
    """Remove card-generated prompt blocks for a fixed experimental catalogue."""
    return re.sub(
        r"\n?# === AUTO-GENERATED START ===.*?# === AUTO-GENERATED END ===\n?",
        "\n",
        text,
        flags=re.DOTALL,
    )

def loadGlobalContext(): #Cat 1

    global GLOBAL_CONTEXT

    system_prompt = ""
    with open("prompts/system_prompt.txt", "r") as file:
        system_prompt = file.read()

    GLOBAL_CONTEXT.append(("system", system_prompt))

    if APPROXIMATION_CONTEXT != 0:

        apx_techniques_summary = ""
        with open("prompts/approximation_techniques.txt", "r") as file:
            apx_techniques_summary = file.read()

        summary_converstaion = [
            (
                "human",
                "Give me a list of code approximations techniques (approximate computing)",
            ),
            ("ai", apx_techniques_summary),
        ]
        GLOBAL_CONTEXT += summary_converstaion

def getPromptTemplates(include_auto_generated_cards=True):
    # Making prompt templates
    prompts = {}

    with open("prompts/target_functions.txt", "r") as file:
        target_functions_script = file.read()

    with open("prompts/purpose_id_vPDG1.txt", "r") as file:
        purpose_identification_script = file.read()

    with open("prompts/annotate_vPDG1.txt", "r") as file:
        annotation_script = file.read()

    with open("prompts/planning_step.txt", "r") as file:
        planning_script = file.read()

    with open("prompts/approximate_vPDG1.txt", "r") as file:
        approximation_script = file.read()

    if not include_auto_generated_cards:
        planning_script = _without_auto_generated_cards(planning_script)
        approximation_script = _without_auto_generated_cards(approximation_script)

    with open("prompts/operator_selection.txt", "r") as file:
        operator_selection_script = file.read()

    with open("prompts/error_handling.txt", "r") as file:
        error_script = file.read()

    with open("prompts/convert_json.txt", "r") as file:
        convert_json_script = file.read()

    prompts['targetFunctionsPrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", target_functions_script),
        ]
    )

    prompts['purposePrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", purpose_identification_script),
        ]
    )

    prompts['annotationPrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", annotation_script),
        ]
    )

    # Per-function operator selection is a plain template rather than a
    # ChatPromptTemplate: it is formatted directly with .format(), matching
    # findApplicableTechniquesForFunction's calling convention.
    prompts['operatorSelectionPrompt'] = operator_selection_script

    prompts['planningPrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", planning_script),
        ]
    )

    prompts['approximationPrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", approximation_script),
        ]
    )

    prompts['errorPrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", error_script),
        ]
    )

    prompts['convertJsonPrompt'] = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{context}"),
            ("human", convert_json_script),
        ]
    )

    return prompts




def loadFormatExamples(): #Cat 1

    global APPROXIMATION_FORMAT_EXAMPLES

    dummy_annotation = []
    dummy_approximation = []
    with open("prompts/FewShotExamples/annotationHuman.txt", "r") as file:
        dummy_annotation.append(formatMessageForHistory(file.read(), False))

    with open("prompts/FewShotExamples/annotationAi.txt", "r") as file:
        dummy_annotation.append(formatMessageForHistory(file.read(), True))

    with open("prompts/FewShotExamples/approximationHuman.txt", "r") as file:
        dummy_approximation.append(formatMessageForHistory(file.read(), False))

    with open("prompts/FewShotExamples/approximationAi.txt", "r") as file:
        dummy_approximation.append(formatMessageForHistory(file.read(), True))

    APPROXIMATION_FORMAT_EXAMPLES = dummy_annotation + dummy_approximation

def loadLoopPerfExamples(): #Cat 1
    with open("prompts/FewShotExamples/loopPerforationExamples.txt",'r') as file:
        LOOP_PERF_EXAMPLES.append(formatMessageForHistory(file.read(),False))
    
    ai_message = "Got it. When applying loop perforation, I'll add to the code in a truncation style approximation to the condition step of the loop."
    LOOP_PERF_EXAMPLES.append(formatMessageForHistory(ai_message,True))

#: ``### Example N: <name> (Technique <id>) ...`` section header.
_EXAMPLE_HEADER = re.compile(
    r"^### Example \d+:.*?\(Technique (\d+)\)", re.MULTILINE
)


def _split_examples_by_technique(text):
    """Return ``(preamble, {technique_id: section_text})``."""
    matches = list(_EXAMPLE_HEADER.finditer(text))
    if not matches:
        return text, {}
    preamble = text[: matches[0].start()]
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[int(match.group(1))] = text[match.start():end]
    return preamble, sections


def loadNewTechniqueExamples(include_auto_generated_cards=True, feasible_ids=None): #Cat 1
    """Load the added-operator few-shot examples.

    All ten examples used to be loaded for every application regardless of
    which operators were feasible, costing roughly 10.6K characters of context
    on every single model call. When ``feasible_ids`` is supplied, only the
    examples for operators this run can actually select are included: a Sobel
    run has no use for the nibble-lookup or link-estimator examples.
    """
    with open("prompts/FewShotExamples/newTechniqueExamples.txt", 'r') as file:
        examples = file.read()

    if not include_auto_generated_cards:
        examples = re.sub(
            r"\n?### === AUTO-GENERATED FEW-SHOT START ===.*?### === AUTO-GENERATED FEW-SHOT END ===\n?",
            "\n",
            examples,
            flags=re.DOTALL,
        )

    included_ids = None
    if feasible_ids:
        preamble, sections = _split_examples_by_technique(examples)
        if sections:
            kept = {
                tid: body for tid, body in sections.items() if tid in set(feasible_ids)
            }
            if kept:
                included_ids = sorted(kept)
                examples = preamble + "".join(kept[tid] for tid in included_ids)
                removed = sorted(set(sections) - set(kept))
                print(
                    f"[context] Few-shot examples: kept T{included_ids} for this "
                    f"target/app, dropped T{removed} "
                    f"({len(examples)} chars instead of "
                    f"{len(preamble) + sum(len(b) for b in sections.values())})"
                )

    NEW_TECHNIQUE_EXAMPLES.append(formatMessageForHistory(examples, False))

    if included_ids:
        listed = ", ".join(f"T{tid}" for tid in included_ids)
        ai_message = (
            f"Got it. I now have implementation examples for {listed}. "
            "I will follow these patterns when applying those operators."
        )
    else:
        ai_message = (
            "Got it. I now have examples of early-exit, spatial downsampling, "
            "temporal decimation, bit-shift EWMA, nibble lookup, radix variation, "
            "pattern segmentation, and lazy preprocessing approximation "
            "techniques. I will follow these patterns when applying these techniques."
        )
    NEW_TECHNIQUE_EXAMPLES.append(formatMessageForHistory(ai_message, True))

def loadCodeBaseSummary(code_summary):
    with open("prompts/manufactured_code_base_summary.txt",'r') as file:
        content = file.read()
        human_prompt = content.format(code_summary= code_summary)
        CODE_BASE_SUMMARY.append(formatMessageForHistory(human_prompt,False))

    ai_message = "Got it! I've added the summary of the functions from the code base to my context. Feel free to ask questions or give further instructions related to this code base."
    CODE_BASE_SUMMARY.append(formatMessageForHistory(ai_message,True))


# Cat 2

def loadTargetFiles(directory): #Cat 2

    global TARGET_FILES

    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith((".c", ".cpp")):
                TARGET_FILES.append(os.path.join(filename))

    return TARGET_FILES

def parseFunctions(): #Cat 2
    for file_name in TARGET_FILES:
        # command = [
        #     "python3",
        #     "compilation_testing/Tools/LSPextractor_all_entities.py",
        #     "target/" + file_name,
        # ]

        lsp_extractor("target/" + file_name,False,False)

        # Execute the command
        # subprocess.run(command, capture_output=True, text=True)
