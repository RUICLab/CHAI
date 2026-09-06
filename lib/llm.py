from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import (
    ChatPromptTemplate,
)

# Lazy imports for HuggingFace (loaded only when API_NAME == 'HuggingFace')
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import dotenv
dotenv.load_dotenv()
import os
import re
import json
from colorama import Fore, Back, Style

from config.globals import (
    ENTITIES,
    GLOBAL_CONTEXT
)

from config.config import (
    PRINT_LLM_CONVO,
    API_NAME,
)

from utils import llm_accounting
from utils import token_budget
from utils.utils import(
    getFunctionData,
    formatMessageForHistory,
    formatConversation,
    writeFunctionsToJson,
    printStructuredJson,
    getCodeBase,
    parseTargetFunctions,
    Dprint,
)

from utils.compiler import(
    compileTest
)
from utils.knob_metadata import drop_fixed_knobs, normalize_knob_metadata
from utils.ollama_config import verify_ollama_context

#  Load the model selected in .env.
model = os.getenv("LLM_MODEL")

# Initialize the language model
llmLangChain = None
if API_NAME == "OpenAI":
    api_key = os.getenv("OPENAI_API_KEY")
    llmLangChain = ChatOpenAI(api_key=api_key, temperature=0, model=model, max_tokens=2048, max_retries=5, timeout=120)
elif API_NAME == "Anthropic":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    llmLangChain = ChatAnthropic(api_key=api_key,model=model, temperature=0, max_tokens=8192)
elif API_NAME == "Ollama":
    # Ollama exposes an OpenAI-compatible chat-completions endpoint. When
    # CHAI runs in Docker, use host.docker.internal to reach the host service.
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "600"))
    max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))
    max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "0"))
    minimum_context = int(os.getenv("OLLAMA_MIN_CONTEXT", "0"))
    configured_context = verify_ollama_context(
        base_url,
        model,
        minimum_context,
        timeout=min(timeout, 10),
        expected_prediction_limit=max_tokens,
    )
    llmLangChain = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
    )
    print(
        f"[Ollama] Using {model} at {base_url} "
        f"(context={configured_context or 'unchecked'}, timeout={timeout:.0f}s, "
        f"max_tokens={max_tokens}, retries={max_retries})"
    )
elif API_NAME == "HuggingFace":
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    import torch

    hf_token = os.getenv("HF_TOKEN")
    print(f"[HuggingFace] Loading {model} onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)

    # Use 4-bit quantization for 70B+ models to fit on 2x RTX 3090 (48GB)
    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    use_4bit = "70B" in model.upper() or "70b" in model
    if use_4bit:
        print(f"[HuggingFace] Using 4-bit NF4 quantization for {model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Balance model across GPUs, leave headroom for KV cache
        n_gpus = torch.cuda.device_count()
        max_mem = {i: "22GiB" for i in range(n_gpus)}
        print(f"[HuggingFace] max_memory = {max_mem}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_mem,
            token=hf_token,
        )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )

    pipe = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.01,            # near-deterministic (0.0 disables sampling)
        do_sample=True,
        return_full_text=False,
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    llmLangChain = ChatHuggingFace(llm=hf_llm)
    print(f"[HuggingFace] {model} loaded successfully.")

print(model)

if llmLangChain == None:
    print("Error no API NAME specified in config.py. Exiting")
    quit()

def purposeIdentificationFunction(
        this_function, 
        this_context, 
        purposePrompt                              
    ):

    global PRINT_LLM_CONVO

    chain = llmLangChain
    function_data = getFunctionData(ENTITIES, this_function)['completeFunction']

    def get_human_prompt_purpose():
        common_prompt = {
            'function_name': this_function,
            'function_code': function_data
        }
        context_prompt = purposePrompt.format(context=this_context, **common_prompt)
        no_context_prompt = purposePrompt.format(**common_prompt)
        return context_prompt, no_context_prompt

    def log_conversation(human_prompt):
        if PRINT_LLM_CONVO:
            print(f"\n\n ---------------------- Purpose Identification Prompt ---------------------- \n\n{human_prompt}\n\n")

    function_purpose = ""
    output_purpose = ""
    MAX_PURPOSE_RETRIES = 5
    purpose_retries = 0

    while True:
        try:    

            human_prompt_purpose, no_context_human_prompt = get_human_prompt_purpose()
            log_conversation(human_prompt_purpose)

            llm_accounting.record("purpose_identification")
            output_purpose = chain.invoke(input=human_prompt_purpose)
            token_budget.record_provider_usage("purpose_identification", output_purpose)

            if PRINT_LLM_CONVO:
                print(Fore.CYAN + str(output_purpose.content))
                print(Style.RESET_ALL)

            function_purpose = output_purpose.content
            break
        except Exception as error:
            purpose_retries += 1
            print(f"[purposeID] Retry {purpose_retries}/{MAX_PURPOSE_RETRIES} for {this_function}: {error}")
            if purpose_retries >= MAX_PURPOSE_RETRIES:
                print(Fore.RED + f"  [purposeID] {this_function} failed after {MAX_PURPOSE_RETRIES} attempts." + Style.RESET_ALL)
                break
            if PRINT_LLM_CONVO:
                print(Fore.RED + str(output_purpose))
                print(Style.RESET_ALL)

    this_purpose_convo = [
        formatMessageForHistory(no_context_human_prompt, False),
        formatMessageForHistory(output_purpose, True)
    ]
    return this_purpose_convo


def annotateFunction(
        this_function, 
        this_context, 
        annotationPrompt,
        output_format_instructions,
        output_format,
    ):

    global PRINT_LLM_CONVO
    # global ANNOTATION_SCRIPT

    chain = annotationPrompt | llmLangChain | output_format
    function_data = getFunctionData(ENTITIES, this_function)['completeFunction']

    def get_human_prompt_anno():        
        # Shared parameters
        common_prompt = {
            'function_name': this_function,
            'function_code': function_data,
            'add_error': this_error,
            'output_format': output_format_instructions,
        }

        # Generate the prompts
        context_prompt = annotationPrompt.format(context=context, **common_prompt)
        no_context_prompt = annotationPrompt.format(**common_prompt)

        return context_prompt, no_context_prompt

    def log_conversation(human_prompt):
        if PRINT_LLM_CONVO:
            print(f"\n\n ----------------------  Annotation Prompt  ---------------------- \n\n{human_prompt}\n\n")

    output_annotation = ""
    this_error = ""
    last_error_convo = []
    MAX_ANNO_RETRIES = 5
    anno_retries = 0

    while True:
        try:    

            # Generate input prompt
            context = this_context + last_error_convo
            human_prompt_anno, no_context_human_prompt = get_human_prompt_anno()
            log_conversation(no_context_human_prompt)

            # Send API call for Annotation
            llm_accounting.record("annotation")
            output_annotation = chain.invoke(input=human_prompt_anno)
            token_budget.record_provider_usage("annotation", output_annotation)

            if PRINT_LLM_CONVO:
                print(Fore.YELLOW + str(output_annotation))
                print(Style.RESET_ALL)

            break
        except Exception as error:
            anno_retries += 1
            print(f"[annotate] Retry {anno_retries}/{MAX_ANNO_RETRIES} for {this_function}: {error}")
            if anno_retries >= MAX_ANNO_RETRIES:
                print(Fore.RED + f"  [annotate] {this_function} failed after {MAX_ANNO_RETRIES} attempts." + Style.RESET_ALL)
                break
            if PRINT_LLM_CONVO:
                print(Fore.RED + str(output_annotation))
                print(Style.RESET_ALL)

            last_error_convo = [
                formatMessageForHistory(no_context_human_prompt, False),
                formatMessageForHistory(output_annotation, True)
            ]
            this_error = f"There was a format error in your previous response. {error}. Make sure to follow the JSON format stated.\n\n"

    this_annotation_convo = [
        formatMessageForHistory(no_context_human_prompt, False),
        formatMessageForHistory(output_annotation, True)
    ]
    return this_annotation_convo


def planStepFunction(
        this_function,
        this_context,
        planningPrompt,
        platform_architecure,
        max_operators=4,
        prev_err="",
    ):

    global PRINT_LLM_CONVO

    chain = llmLangChain

    this_error = prev_err

    def get_human_prompt_planing(context):
        # Cache function data

        # Shared parameters
        common_prompt = {
            'function_name': this_function,
            'platform_architecture': platform_architecure,
            'max_operators': max_operators,
            'add_error': this_error,
        }

        # Generate the prompts
        context_prompt = planningPrompt.format(context=context, **common_prompt)
        no_context_prompt = planningPrompt.format(**common_prompt)

        return context_prompt, no_context_prompt

    def log_conversation(human_prompt):
        if PRINT_LLM_CONVO:
            print(f"\n\n ----------------------  Planning Prompt  ---------------------- \n\n{human_prompt}\n\n")

    output_plan = ""
    last_error_convo = []
    MAX_PLAN_RETRIES = 5
    plan_retries = 0

    while True:
        try:    

            # Generate input prompt
            context = this_context + last_error_convo
            human_prompt, no_context_human_prompt = get_human_prompt_planing(context)
            log_conversation(no_context_human_prompt)

            # Send API call for Planning step
            llm_accounting.record("planning")
            output_plan = chain.invoke(input=human_prompt)
            token_budget.record_provider_usage("planning", output_plan)

            if PRINT_LLM_CONVO:
                print(Fore.YELLOW + str(output_plan.content))
                print(Style.RESET_ALL)

            break
        except Exception as error:
            plan_retries += 1
            print(f"[planning] Retry {plan_retries}/{MAX_PLAN_RETRIES} for {this_function}: {error}")
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if plan_retries >= MAX_PLAN_RETRIES:
                print(Fore.RED + f"  [planning] {this_function} failed after {MAX_PLAN_RETRIES} attempts." + Style.RESET_ALL)
                break
            _plan_content = output_plan.content if hasattr(output_plan, 'content') else str(output_plan)
            if PRINT_LLM_CONVO:
                print(Fore.RED + _plan_content)
                print(Style.RESET_ALL)

            last_error_convo = [
                formatMessageForHistory(no_context_human_prompt, False),
                formatMessageForHistory(_plan_content, True)
            ]
            this_error = f"There was a format error in your previous response. {error}. Make sure to follow the JSON format stated.\n\n"

    _plan_content_final = output_plan.content if hasattr(output_plan, 'content') else str(output_plan)
    this_convo = [
        formatMessageForHistory(no_context_human_prompt, False),
        formatMessageForHistory(_plan_content_final, True)
    ]
    return this_convo

def approximateFunction(
        this_function,
        this_context, 
        approximation_prompt,
        prev_err
    ):

    global PRINT_LLM_CONVO
    Dprint("LLL: ",PRINT_LLM_CONVO)
    # global APPROXIMATION_SCRIPT
    chain = llmLangChain

    def get_human_prompt_approx(context):
        common_prompt = {
            "function_name": this_function,
            "add_error": this_error,
        }
        context_prompt = approximation_prompt.format(context=context, **common_prompt)
        no_context_prompt = approximation_prompt.format(**common_prompt)
        return context_prompt, no_context_prompt

    def log_conversation(human_prompt):
        if PRINT_LLM_CONVO:
            print(f"\n\n ---------------------- Approximation Prompt ---------------------- \n\n{human_prompt}\n\n")


    last_error = []
    this_error = prev_err
    output_approximated = ""
    MAX_APPROX_RETRIES = 5
    approx_retries = 0

    while True:
        try:
            context = this_context + last_error
            human_prompt_approx, no_context_human_prompt = get_human_prompt_approx(context)
            log_conversation(human_prompt_approx)

            llm_accounting.record("approximation")
            output_approximated = chain.invoke(input=human_prompt_approx)
            token_budget.record_provider_usage("approximation", output_approximated)

            if PRINT_LLM_CONVO:
                print(Fore.GREEN + str(output_approximated.content))
                print(Style.RESET_ALL)

            # Compilation error check handeling
            break

        except Exception as error:
            approx_retries += 1
            print(f"[approximateFunction] Retry {approx_retries}/{MAX_APPROX_RETRIES} for {this_function}: {error}")
            import gc; gc.collect()
            if approx_retries >= MAX_APPROX_RETRIES:
                print(Fore.RED + f"  [approximateFunction] {this_function} failed after {MAX_APPROX_RETRIES} attempts — skipping." + Style.RESET_ALL)
                break

            if PRINT_LLM_CONVO:
                print(Fore.YELLOW + str(output_approximated))
                print(Style.RESET_ALL)

            # Guard: output_approximated may still be a plain string if OOM hit before any response
            _approx_content = output_approximated.content if hasattr(output_approximated, 'content') else str(output_approximated)
            last_error = [
                formatMessageForHistory(no_context_human_prompt, False),
                formatMessageForHistory(_approx_content, True)
            ]
            this_error = f"There was a format error in your previous response. {error}. Make sure to follow the JSON format stated.\n\n"

    _approx_content_final = output_approximated.content if hasattr(output_approximated, 'content') else str(output_approximated)
    this_convo = [
        formatMessageForHistory(no_context_human_prompt, False),
        formatMessageForHistory(_approx_content_final, True)
    ]
    return this_convo


def convertJson(
        this_function,
        this_context,
        convert_json_prompt,
        output_format_parser
    ):

    global PRINT_LLM_CONVO
    # Use raw LLM (no piped parser) so we can manually extract JSON from 8B model output
    chain_with_parser = llmLangChain | output_format_parser
    chain_raw = llmLangChain
    this_error = ""

    import re as _re
    import json as _json
    import ast as _ast

    def _extract_json_from_text(text):
        """Try to extract a JSON object from LLM text that may contain markdown/extra text.
        Handles both JSON and Python dict formats, and maps 'approximated_code' to 'approxmated_code'.
        Also normalizes value types (JSON arrays → strings) to match Pydantic model expectations."""
        
        def _normalize_keys(d):
            """Map correctly-spelled key to the misspelled key the codebase expects."""
            if isinstance(d, dict) and 'approximated_code' in d and 'approxmated_code' not in d:
                d['approxmated_code'] = d.pop('approximated_code')
            return d

        def _normalize_values(d):
            """Convert JSON arrays/objects in knob fields to the string format the codebase expects.
            The Pydantic model declares knob_variables, knob_ranges, knob_increments as str fields,
            so if the LLM returns actual lists, convert them."""
            for key in ['knob_variables', 'knob_ranges', 'knob_increments']:
                if key in d and isinstance(d[key], (list, dict)):
                    d[key] = str(d[key])
            return d

        def _find_outermost_braces(s):
            """Find the outermost { ... } respecting quotes (single and double).
            Returns the substring including outer braces, or None."""
            depth = 0
            start = -1
            in_str = False
            str_char = None
            i = 0
            while i < len(s):
                c = s[i]
                if in_str:
                    if c == '\\':
                        i += 2  # skip escaped char
                        continue
                    if c == str_char:
                        in_str = False
                elif c in ('"', "'"):
                    in_str = True
                    str_char = c
                elif c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        return s[start:i+1]
                i += 1
            return None

        def _try_parse(fragment):
            """Try json.loads then ast.literal_eval on a text fragment."""
            for parser in [_json.loads, _ast.literal_eval]:
                try:
                    result = parser(fragment)
                    if isinstance(result, dict):
                        return _normalize_values(_normalize_keys(result))
                except Exception:
                    pass
            return None

        # Strategy 1: Direct JSON/Python parse
        result = _try_parse(text)
        if result:
            return result

        # Strategy 2: Extract from markdown code fences (```json ... ``` or ``` ... ```)
        # Use greedy .* between fences — the content between fences IS the JSON
        fence_match = _re.search(r'```(?:json)?\s*(.*?)\s*```', text, _re.DOTALL)
        if fence_match:
            fenced_content = fence_match.group(1)
            # Find outermost braces within fenced content (handles nested {} in C code)
            json_fragment = _find_outermost_braces(fenced_content)
            if json_fragment:
                result = _try_parse(json_fragment)
                if result:
                    return result
            # Also try the whole fenced content directly
            result = _try_parse(fenced_content)
            if result:
                return result

        # Strategy 3: Find outermost { ... } in the entire text
        json_fragment = _find_outermost_braces(text)
        if json_fragment:
            result = _try_parse(json_fragment)
            if result:
                return result

        # Strategy 4: Robust key-value extraction for badly-formatted output
        # Handles embedded newlines, braces in C code, etc.
        code_key = None
        for k in ['approxmated_code', 'approximated_code']:
            if k in text:
                code_key = k
                break
        
        if code_key:
            try:
                dict_start = text.find('{')
                if dict_start >= 0:
                    result = {}
                    keys_to_find = [code_key, 'knob_variables', 'knob_ranges', 'knob_increments', 'discription']
                    
                    for i, key in enumerate(keys_to_find):
                        key_patterns = ["'" + key + "'", '"' + key + '"']
                        key_pos = -1
                        for kp in key_patterns:
                            pos = text.find(kp, dict_start)
                            if pos >= 0:
                                key_pos = pos
                                break
                        
                        if key_pos < 0:
                            continue
                        
                        colon_pos = text.find(':', key_pos + len(key))
                        if colon_pos < 0:
                            continue
                        
                        # Skip whitespace after colon
                        val_cursor = colon_pos + 1
                        while val_cursor < len(text) and text[val_cursor] in (' ', '\t', '\n', '\r'):
                            val_cursor += 1
                        
                        if val_cursor >= len(text):
                            continue

                        # Value can be a string (quoted) or an array/object
                        if text[val_cursor] in ("'", '"'):
                            quote_char = text[val_cursor]
                            # Find the next key or end to bound our search
                            next_key_pos = len(text)
                            for next_key in keys_to_find[i+1:]:
                                for kp in ["'" + next_key + "'", '"' + next_key + '"']:
                                    pos = text.find(kp, val_cursor + 1)
                                    if 0 <= pos < next_key_pos:
                                        next_key_pos = pos
                            # Search backwards for closing quote
                            val_end = -1
                            for j in range(min(next_key_pos, len(text)) - 1, val_cursor, -1):
                                if text[j] == quote_char:
                                    val_end = j
                                    break
                            if val_end > val_cursor:
                                result[key] = text[val_cursor + 1:val_end]
                        elif text[val_cursor] == '[':
                            # Array value — find matching ]
                            bracket_depth = 0
                            for j in range(val_cursor, len(text)):
                                if text[j] == '[':
                                    bracket_depth += 1
                                elif text[j] == ']':
                                    bracket_depth -= 1
                                    if bracket_depth == 0:
                                        arr_text = text[val_cursor:j+1]
                                        try:
                                            result[key] = str(_json.loads(arr_text))
                                        except Exception:
                                            result[key] = arr_text
                                        break
                    
                    if code_key in result:
                        return _normalize_keys(result)
            except Exception:
                pass

        return None

    def get_human_prompt_json(context):
        common_prompt = {
            'add_error': this_error,
            'output_instuctions': output_format_parser.get_format_instructions()
        }

        context_prompt = convert_json_prompt.format(context=context, **common_prompt)
        no_context_prompt = convert_json_prompt.format(**common_prompt)

        return context_prompt, no_context_prompt
    
    def log_conversation(human_prompt):
        if PRINT_LLM_CONVO:
            print(f"\n\n ---------------------- JSON Conversion Prompt ---------------------- \n\n{human_prompt}\n\n")

    def extract_or_default(output, key, default='[]'):
        return output.get(key, default)

    last_error = []
    this_error = ""
    output_approximated = ""
    MAX_JSON_RETRIES = 5
    json_retries = 0
    approximate_function = None

    while True:
        try:
            context = this_context + last_error
            human_prompt_json, no_context_prompt = get_human_prompt_json(context)
            log_conversation(no_context_prompt)

            # Always get raw LLM response first (single call)
            llm_accounting.record("json_conversion")
            raw_response = chain_raw.invoke(input=human_prompt_json)
            token_budget.record_provider_usage("json_conversion", raw_response)
            raw_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

            # Try parsing with output_format_parser
            parsed = None
            try:
                parsed = output_format_parser.parse(raw_text)
                if isinstance(parsed, dict):
                    output_approximated = parsed
                else:
                    output_approximated = parsed.dict() if hasattr(parsed, 'dict') else dict(parsed)
            except Exception:
                # Fallback: manually extract JSON/Python-dict from raw text
                parsed = _extract_json_from_text(raw_text)
                if parsed and isinstance(parsed, dict) and 'approxmated_code' in parsed:
                    output_approximated = parsed
                    print(f"[convertJson] Manual extraction succeeded for {this_function}")
                else:
                    raise KeyError(f"Could not extract JSON from LLM output (first 300 chars): {raw_text[:300]}")

            if PRINT_LLM_CONVO:
                print(Fore.GREEN + str(output_approximated))
                print(Style.RESET_ALL)

            function_code_approximated = output_approximated['approxmated_code']
            knobs_variables_list = extract_or_default(output_approximated, 'knob_variables')
            knobs_variables_ranges = extract_or_default(output_approximated, 'knob_ranges')
            knobs_variables_step_size = extract_or_default(output_approximated, 'knob_increments')

            approximate_function = {
                "functionName": this_function,
                "completeFunction": function_code_approximated,
                "knobVariables": knobs_variables_list,
                "knobRanges": knobs_variables_ranges,
                "knobStepSize": knobs_variables_step_size,
            }

            fixed_knobs = drop_fixed_knobs(approximate_function, mutate=True)
            if fixed_knobs:
                print(
                    f"[convertJson] Kept fixed local constant(s) but removed "
                    f"non-tunable metadata for {this_function}: {fixed_knobs}"
                )

            # Reject symbolic, mismatched, or otherwise unusable BO metadata
            # while the LLM is still in its JSON repair loop.
            normalize_knob_metadata(approximate_function, mutate=True)

            break

        except Exception as error:
            json_retries += 1
            print(f"[convertJson] Retry {json_retries}/{MAX_JSON_RETRIES} for {this_function}: {error}")
            if json_retries >= MAX_JSON_RETRIES:
                print(Fore.RED + f"  [convertJson] {this_function} failed JSON parsing after {MAX_JSON_RETRIES} attempts — returning empty." + Style.RESET_ALL)
                approximate_function = {
                    "functionName": this_function,
                    "completeFunction": "",
                    "knobVariables": "[]",
                    "knobRanges": "[]",
                    "knobStepSize": "[]",
                }
                break

            if PRINT_LLM_CONVO:
                print(Fore.YELLOW + str(output_approximated))
                print(Style.RESET_ALL)

            last_error = [
                formatMessageForHistory(no_context_prompt, False),
                formatMessageForHistory(output_approximated, True)
            ]
            this_error = f"There was a format error in your previous response. {error}. Make sure to follow the JSON format stated.\n\n"

    this_convo = [
        formatMessageForHistory(no_context_prompt, False),
        formatMessageForHistory(output_approximated, True)
    ]
    return this_convo, approximate_function

def approximateFunctionOLD(
        approximated_functions_dict, 
        this_function, 
        this_context, 
        approximation_prompt,
        output_parser, 
        function_code_annotated, 
        output_format_instructions_apx
    ):

    global PRINT_LLM_CONVO
    # global APPROXIMATION_SCRIPT
    chain = approximation_prompt | llmLangChain | output_parser

    def get_human_prompt_approx():
        return approximation_prompt.format(
            annotated_code=function_code_annotated,
            add_error=this_error,
            output_format=output_format_instructions_apx
        )

    def log_conversation(context, human_prompt):
        if PRINT_LLM_CONVO:
            to_print_prompt = formatConversation(context + [formatMessageForHistory(human_prompt, False)])
            Dprint(f"\n\n ---------------------- Approximation Prompt ---------------------- \n\n{to_print_prompt}\n\n")

    def extract_or_default(output, key, default='[]'):
        return output.get(key, default)

    last_error = []
    this_error = ""
    output_approximated = ""
    MAX_OLD_RETRIES = 5
    old_retries = 0

    while True:
        try:
            context = this_context + last_error
            human_prompt_approx = get_human_prompt_approx()
            log_conversation(context, human_prompt_approx)

            llm_accounting.record("approximation_legacy")
            output_approximated = chain.invoke(input={
                'output_format': "\n The output format should be in JSON \n" + output_format_instructions_apx,
                'annotated_code': function_code_annotated,
                'add_error': this_error,
                'context': context
            })

            if PRINT_LLM_CONVO:
                Dprint(Fore.GREEN)
                printStructuredJson(output_approximated)
                Dprint(Style.RESET_ALL)

            function_code_approximated = output_approximated['approxmated_code']
            knobs_variables_list = extract_or_default(output_approximated, 'knob_variables')
            knobs_variables_ranges = extract_or_default(output_approximated, 'knob_ranges')
            knobs_variables_step_size = extract_or_default(output_approximated, 'knob_increments')

            approximated_functions_dict[this_function] = {
                "functionName": this_function,
                "completeFunction": function_code_approximated,
                "knobVariables": knobs_variables_list,
                "knobRanges": knobs_variables_ranges,
                "knobStepSize": knobs_variables_step_size,
            }

            writeFunctionsToJson(approximated_functions_dict, 'approximated_functions/apx')

            compile_error = compileTest(function=this_function)
            if compile_error:
                old_retries += 1
                if old_retries >= MAX_OLD_RETRIES:
                    print(Fore.RED + f"  [approximateFunctionOLD] {this_function} failed after {MAX_OLD_RETRIES} attempts — skipping." + Style.RESET_ALL)
                    break
                last_error = [
                    formatMessageForHistory(human_prompt_approx, False),
                    formatMessageForHistory(output_approximated, True)
                ]
                this_error = f"This error occurred at compile time: \n{compile_error}\n"
                continue

            break

        except Exception as error:
            old_retries += 1
            print(f"[approximateFunctionOLD] Retry {old_retries}/{MAX_OLD_RETRIES} for {this_function}: {error}")
            if old_retries >= MAX_OLD_RETRIES:
                print(Fore.RED + f"  [approximateFunctionOLD] {this_function} failed after {MAX_OLD_RETRIES} attempts — skipping." + Style.RESET_ALL)
                break

            if PRINT_LLM_CONVO:
                Dprint(Fore.YELLOW + str(output_approximated))
                Dprint(Style.RESET_ALL)

            last_error = [
                formatMessageForHistory(human_prompt_approx, False),
                formatMessageForHistory(output_approximated, True)
            ]
            this_error = f"There was a format error in your previous response. {error}. Make sure to follow the JSON format stated.\n\n"

    return output_approximated, human_prompt_approx




rtl_chat_histroy = []


def _strip_markdown_code_fence(text: str) -> str:
    """Return code-only output when an otherwise valid response is fenced."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"

def generateExpandMakeFile(
        files,
        compiler_error=""
    ):
    

    print("Sending PDG Makefile prompt...")


    next_prompt = ""
    with open('prompts/compilerRTL.txt','r') as file: # Move reading file to init file
        next_prompt = file.read()

    prompt = ChatPromptTemplate.from_messages([
    ('placeholder', '{conversation}'),
    ('human', next_prompt)
    ])
    
    print("Sending PDG Makefile prompt...")
    chain = prompt | llmLangChain
    llm_accounting.record("makefile_rtl")
    output = chain.invoke(input={'conversation':rtl_chat_histroy,'compiler_error':compiler_error,'files_list':files})
    print("Generated PDG Makefile")


    rtl_chat_histroy.append(formatMessageForHistory(prompt.format(files_list=files,compiler_error=compiler_error),False))
    rtl_chat_histroy.append(formatMessageForHistory(output.content,True))
    
    print("\n\n\n ------ Prompt: ------ \n\n")
    print(prompt.format(files_list=files,compiler_error=compiler_error,conversation=rtl_chat_histroy))
    print("\n\n\n ------ LLM Responce: ------ \n\n")
    print(output.content)
    return _strip_markdown_code_fence(output.content)

def _cached_target_function_fallback(app_name):
    """Recover known target functions when a local model emits no valid map."""
    if not app_name:
        return {}
    path = os.path.join(
        "llm-prerun", app_name, "approximated_functions", "apx_all.json"
    )
    try:
        with open(path, "r") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    targets = {}
    for entry in entries if isinstance(entries, list) else []:
        name = entry.get("functionName")
        if name:
            targets[name] = "approximate"
    return targets


def findTargetFunctions(function_target_prompt, fallback_app_name=None):

    global PRINT_LLM_CONVO
    global GLOBAL_CONTEXT

    # Step 0: Check if 'code_summary' and 'target_functions' are present in 'logs/somefile.txt'
    file_path = 'logs/code_base_info.txt'
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            if 'code_summary' in data and 'target_functions' in data:
                print("Loading cached data from logs/code_base_info.txt")
                return data['target_functions'], data['code_summary']


    # Step 1: Gather context and codebase
    context = GLOBAL_CONTEXT.copy()
    code_base = getCodeBase()

    # Step 2: Format the initial prompt with the provided context and codebase
    formatted_prompt = function_target_prompt.format(
        context=context,
        code_base=code_base,
    )

    # Step 3: Invoke LLM for target functions
    output = invokeLLM(formatted_prompt, llmLangChain, stage="target_selection")
    if PRINT_LLM_CONVO:
        print("--- Target_function: ---\n \n")
        print(output)   

    # Step 4: Parse the target functions from the LLM's output
    target_functions = parseTargetFunctions(output) if output else None
    has_approximation = (
        isinstance(target_functions, dict)
        and any(str(choice).lower() == "approximate"
                for choice in target_functions.values())
    )
    if not has_approximation:
        fallback_targets = _cached_target_function_fallback(fallback_app_name)
        if fallback_targets:
            print(
                "[target selection] Local-model response contained no usable "
                f"approximation target; using cached targets for {fallback_app_name}: "
                f"{sorted(fallback_targets)}"
            )
            target_functions = fallback_targets
        else:
            print("[target selection] No approximation target was selected.")
            target_functions = {}

    # Step 5: Update context with the function target prompt and LLM's output
    context += formatMessageForHistory(function_target_prompt.format(code_base=code_base), False)
    context += formatMessageForHistory(output, True)

    # Step 6: Read the code summary prompt from an external file
    with open('prompts/code_summary.txt', 'r') as file: # Move read to init file
        summary_prompt = file.read()

    # Step 7: Create a chat prompt template using the updated context and code summary prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        ('placeholder', '{context}'),
        ('human', summary_prompt)
    ])

    # Step 8: Format the prompt with the current context
    final_prompt = chat_prompt.format(
        context=context
    )

    # Step 9: Invoke LLM to get a code summary
    code_summary = invokeLLM(final_prompt, llmLangChain, stage="code_summary")
    if PRINT_LLM_CONVO:
        print("--- code_summary: ---\n \n")
        print(code_summary)

    # Step 10: Save 'code_summary' and 'target_functions' to 'logs/code_base_info.txt'
    data_to_save = {
        'code_summary': code_summary,
        'target_functions': target_functions,
        'target_selection_raw': output,
    }
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file)

    # Step 11: Return the parsed target functions and code summary
    return target_functions, code_summary


def invokeLLM(input, invoking_obj, stage="uncategorized"):

    output = ""
    try:
        llm_accounting.record(stage)
        output = invoking_obj.invoke(input=(input))
        token_budget.record_provider_usage(stage, output)
    except Exception as error:
        print("While attempting to invoke LLM encoundered error: ",error)
        return None
    
    return output.content
    
    
def findApplicableTechniquesForFunction(
        this_function,
        this_context,
        feasible_techniques,
        techniqueSelectionPrompt,
        max_operators=4,
        ranked_fallback=None,
    ):
    """
    Ask the LLM which of the hardware-feasible techniques are plausible
    candidates for THIS specific function (not the whole app). Mirrors
    purposeIdentificationFunction's calling convention exactly.

    Parameters
    ----------
    this_function : str
    this_context : list  (conversation so far — e.g. purpose ID convo)
    feasible_techniques : dict[int, TechniqueEntry]
        The app-wide hardware-feasible set (already filtered for
        FPU/SIMD/RAM). This function narrows it further, it does not
        re-check hardware feasibility.
    techniqueSelectionPrompt : str
        Template with {context}, {function_name}, {function_code},
        {feasible_techniques} placeholders (same style as purposePrompt).

    Returns
    -------
    this_tech_select_convo : list
        [human_msg, ai_msg] to append to this_context / CHAT_HISTORY.
    candidate_ids : set[int]
        Subset of feasible_techniques.keys() the LLM judged applicable.
        Falls back to the FULL feasible_techniques key set if parsing
        fails or the LLM returns nothing usable (fail-open, matches the
        existing hw_validator fallback philosophy of never silently
        producing an empty prompt).
    """
    from lib.constraint_gatekeeper import format_feasible_list
    from lib.hw_validator import extract_technique_ids_from_text

    global PRINT_LLM_CONVO

    chain = llmLangChain
    function_data = getFunctionData(ENTITIES, this_function)['completeFunction']
    feasible_list_text = format_feasible_list(feasible_techniques)

    def get_human_prompt_tech_select():
        common_prompt = {
            'function_name': this_function,
            'function_code': function_data,
            'feasible_techniques': feasible_list_text,
            'max_operators': max_operators,
        }
        context_prompt = techniqueSelectionPrompt.format(context=this_context, **common_prompt)
        no_context_prompt = techniqueSelectionPrompt.format(context="", **common_prompt)
        return context_prompt, no_context_prompt

    def log_conversation(human_prompt):
        if PRINT_LLM_CONVO:
            print(f"\n\n ---------------------- Technique Selection Prompt ---------------------- \n\n{human_prompt}\n\n")

    output_tech_select = ""
    MAX_TECH_SELECT_RETRIES = 5
    tech_select_retries = 0

    while True:
        try:
            human_prompt_tech_select, no_context_human_prompt = get_human_prompt_tech_select()
            log_conversation(human_prompt_tech_select)

            llm_accounting.record("operator_selection")
            output_tech_select = chain.invoke(input=human_prompt_tech_select)
            token_budget.record_provider_usage("operator_selection", output_tech_select)

            if PRINT_LLM_CONVO:
                print(Fore.CYAN + str(output_tech_select.content))
                print(Style.RESET_ALL)

            break
        except Exception as error:
            tech_select_retries += 1
            print(f"[techSelect] Retry {tech_select_retries}/{MAX_TECH_SELECT_RETRIES} for {this_function}: {error}")
            if tech_select_retries >= MAX_TECH_SELECT_RETRIES:
                print(Fore.RED + f"  [techSelect] {this_function} failed after {MAX_TECH_SELECT_RETRIES} attempts — "
                      f"falling back to full feasible set." + Style.RESET_ALL)
                break
            if PRINT_LLM_CONVO:
                print(Fore.RED + str(output_tech_select))
                print(Style.RESET_ALL)

    response_text = output_tech_select.content if hasattr(output_tech_select, 'content') else str(output_tech_select)

    def this_tech_select_convo_from(text):
        return [
            formatMessageForHistory(no_context_human_prompt, False),
            formatMessageForHistory(text, True),
        ]

    candidate_ids = extract_technique_ids_from_text(response_text) if response_text else set()
    # Guard: LLM must only pick from what's actually hardware-feasible;
    # anything else it names gets dropped, not trusted.
    candidate_ids = candidate_ids & set(feasible_techniques.keys())

    declined = bool(response_text) and re.search(
        r"SELECTED_OPERATORS\s*:\s*none", response_text, re.IGNORECASE
    )
    if declined:
        print(
            Fore.CYAN
            + f"  [techSelect] {this_function}: model reports no applicable operator."
            + Style.RESET_ALL
        )
        return this_tech_select_convo_from(response_text), set()

    # Bounded fallback. Falling back to the *complete* feasible set turned this
    # narrowing step into a no-op, which is how unrelated operators reached
    # every function's plan. Fall back to the cheapest ranked operators
    # instead, so an unparseable answer still narrows the search space.
    if not candidate_ids:
        if ranked_fallback:
            candidate_ids = {
                tid for tid, *_ in ranked_fallback[:max_operators]
            } & set(feasible_techniques.keys())
        if not candidate_ids:
            candidate_ids = set(sorted(feasible_techniques.keys())[:max_operators])
        print(
            Fore.YELLOW
            + f"  [techSelect] {this_function}: no valid selection parsed — "
              f"narrowing to {len(candidate_ids)} cheapest feasible operators "
              f"{sorted(candidate_ids)} instead of the full set "
              f"({len(feasible_techniques)})."
            + Style.RESET_ALL
        )

    # Cap dimensionality even when the model over-selects.
    if len(candidate_ids) > max_operators:
        order = [tid for tid, *_ in (ranked_fallback or [])]
        ordered = sorted(
            candidate_ids,
            key=lambda tid: order.index(tid) if tid in order else len(order),
        )
        dropped = set(ordered[max_operators:])
        candidate_ids = set(ordered[:max_operators])
        print(
            Fore.YELLOW
            + f"  [techSelect] {this_function}: capped selection at "
              f"{max_operators} operators; dropped {sorted(dropped)}."
            + Style.RESET_ALL
        )

    return this_tech_select_convo_from(response_text), candidate_ids
