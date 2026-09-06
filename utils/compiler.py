import subprocess
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import shutil

from utils import llm_accounting

from config.config import(
    API_NAME,
)
from utils.utils import (
    copyFiles,
    Dprint
)
from utils.data_structures import (
    FixedSizePairList
)
from lib.lsp import(
    lsp_patcher
)

load_dotenv()

def queryGPT(chat, chat_history, files_list, error_message=""): 

    # Move the prompt to init file
    """
    Query GPT to generate a Makefile based on the given file list and error message.

    :param chat: ChatOpenAI instance for communication
    :param chat_history: History of chat messages
    :param files_list: List of files in the target directory
    :param error_message: Error message from the previous compilation attempt
    :return: Generated Makefile content
    """
    query_template = """
    Given a list of files in the directory, output a Makefile to compile the application.

    Files: files = {files_list}

    Output only the Makefile content without any additional text. Your exact output will be directly pasted into the Makefile (so do not include any formatting like "```").

    The command that will be run is simply make main. Do not include the -Werror or -Wall flags. Make sure to include -DLOCAL_RUN flag.
    """

    prompt = ChatPromptTemplate.from_messages(
        [("placeholder", "{conversation}"), ("human", "{next_prompt}")]
    )

    query = query_template.format(files_list=files_list) + "\n" + error_message

    chain = prompt | chat

    import time as _time
    _max_retries = 5
    for _attempt in range(1, _max_retries + 1):
        try:
            llm_accounting.record(
                "makefile_generation" if _attempt == 1
                else "makefile_generation_retry"
            )
            response = chain.invoke(
                input={"conversation": list(chat_history), "next_prompt": query}
            )
            break
        except Exception as _e:
            _err_str = str(_e).lower()
            if '429' in _err_str or 'rate' in _err_str:
                _wait = 60 * _attempt
                Dprint(f"[queryGPT] Rate limit hit, waiting {_wait}s (attempt {_attempt}/{_max_retries})...")
                _time.sleep(_wait)
            else:
                Dprint(f"[queryGPT] Error (attempt {_attempt}/{_max_retries}): {_e}")
                _time.sleep(5)
            if _attempt == _max_retries:
                raise

    chat_history.add(formatMessageForHistory(query, False))
    chat_history.add(formatMessageForHistory(response.content, True))

    return response.content

def formatMessageForHistory(content, is_ai):
    """
    Format a message for chat history.

    :param content: Content of the message
    :param is_ai: Boolean indicating if the message is from AI
    :return: Tuple representing the message
    """
    return ("ai" if is_ai else "human", content)

def getFilesList(directory):
    """
    Get a list of files in the specified directory.

    :param directory: Target directory path
    :return: List of filenames in the directory
    """
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(filename)
    return files

def compileAndLog(target_path):
    """
    Attempt to compile the project and log the output.

    :param target_path: Path to the target directory
    :return: Tuple indicating success (bool) and error message (str)
    """
    try:
        with open(os.path.join(target_path, "compiler_log.txt"), "w") as output_file:
            # A compilation-test directory is copied from ``target`` and can
            # therefore contain a binary/object produced by an earlier build.
            # Without cleaning first, ``make main`` may report that stale
            # artifact as up to date and never compile the newly patched C
            # function.  Generation and BayesOpt must use the same clean-build
            # gate or an invalid plan can pass generation and fail only later.
            subprocess.run(
                "make clean",
                shell=True,
                check=True,
                cwd=target_path,
                stdout=output_file,
                stderr=subprocess.STDOUT,
            )
            subprocess.run(
                "make main",
                shell=True,
                check=True,
                cwd=target_path,
                stdout=output_file,
                stderr=subprocess.STDOUT,
            )
        return True, ""
    except subprocess.CalledProcessError as e:
        with open(os.path.join(target_path, "compiler_log.txt"), "r") as output_file:
            return False, output_file.read()

def generateMakeFile(target_path):

    global API_NAME

    chat = None
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")
    if API_NAME == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        chat = ChatOpenAI(api_key=api_key, temperature=0.7, model=model, max_retries=5, timeout=120)
    elif API_NAME == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        chat = ChatAnthropic(api_key=api_key,model=model)
    elif API_NAME == "Ollama":
        chat = ChatOpenAI(
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            temperature=0,
            model=model,
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "0")),
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "600")),
        )
    elif API_NAME == "HuggingFace":
        # Re-use the global LLM from lib/llm.py to avoid loading model twice
        from lib.llm import llmLangChain
        chat = llmLangChain

    if chat is None:
        Dprint("Error: No LLM configured for compilation. Check API_NAME in config.py.")
        return 1

    chat_history = FixedSizePairList(5)

    files_list = getFilesList(target_path)

    response = queryGPT(chat, chat_history, files_list)

    makefile_path = os.path.join(target_path, "Makefile")

    def _fix_makefile_tabs(content):
        """Ensure Makefile recipe lines use tabs, not spaces.
        A recipe line is any line after a target rule (line ending with :) that starts with whitespace."""
        import re as _re
        # Strip markdown fences if LLM wraps output
        content = _re.sub(r'^```(?:makefile|make)?\s*\n', '', content, flags=_re.MULTILINE)
        content = _re.sub(r'\n```\s*$', '', content)
        # Replace leading spaces (4 or 2 spaces) with a tab on recipe lines
        lines = content.split('\n')
        fixed = []
        in_recipe = False
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#') and ':' in stripped and not stripped.startswith('\t') and '=' not in stripped.split(':')[0]:
                # This looks like a target rule
                in_recipe = True
                fixed.append(line)
            elif in_recipe and line and line[0] in (' ', '\t'):
                # Recipe line — ensure it starts with tab
                fixed.append('\t' + stripped)
            else:
                if stripped == '':
                    in_recipe = False
                fixed.append(line)
        return '\n'.join(fixed)

    def _strip_missing_deps(content, directory):
        """Remove dependencies on files that don't exist in the target directory."""
        import re as _re
        existing_files = set(os.listdir(directory))
        lines = content.split('\n')
        fixed = []
        for line in lines:
            # Match target rule lines like "main.o: main.c header.h"
            m = _re.match(r'^(\S+)\s*:\s*(.+)$', line)
            if m and '=' not in line:
                target = m.group(1)
                deps = m.group(2).split()
                # Keep only deps that exist or are variables/patterns
                valid_deps = [d for d in deps if d in existing_files or '$' in d or '%' in d]
                if valid_deps:
                    fixed.append(f"{target}: {' '.join(valid_deps)}")
                else:
                    fixed.append(f"{target}:")
            else:
                fixed.append(line)
        return '\n'.join(fixed)

    def _generate_fallback_makefile(directory):
        """Generate a simple, reliable Makefile based on actual source files present."""
        import glob
        c_files = sorted(glob.glob(os.path.join(directory, '*.c')))
        if not c_files:
            return None
        c_basenames = [os.path.basename(f) for f in c_files]
        o_basenames = [f.replace('.c', '.o') for f in c_basenames]
        
        lines = [
            'CC=gcc',
            'CFLAGS=-c -O2 -DLOCAL_RUN',
            '',
        ]
        # Individual .o rules
        for c, o in zip(c_basenames, o_basenames):
            lines.append(f'{o}: {c}')
            lines.append(f'\t$(CC) $(CFLAGS) {c} -o {o}')
            lines.append('')
        
        # Link rule
        lines.append(f"main: {' '.join(o_basenames)}")
        lines.append(f"\t$(CC) {' '.join(o_basenames)} -o main -lm")
        lines.append('')
        lines.append('clean:')
        lines.append(f"\trm -f {' '.join(o_basenames)} main")
        lines.append('')
        return '\n'.join(lines)

    for _ in range(10):
        makefile_content = _fix_makefile_tabs(response)
        makefile_content = _strip_missing_deps(makefile_content, target_path)
        with open(makefile_path, "w") as file:
            file.write(makefile_content)

        success, error_message = compileAndLog(target_path)
        
        if success:
            Dprint("Compilation successful\n")
            return 0

        response = queryGPT(chat, chat_history, files_list, error_message)

    # All LLM attempts failed — try a deterministic fallback Makefile
    Dprint("LLM Makefile generation failed after 10 attempts. Trying fallback Makefile...\n")
    fallback = _generate_fallback_makefile(target_path)
    if fallback:
        with open(makefile_path, "w") as file:
            file.write(fallback)
        success, error_message = compileAndLog(target_path)
        if success:
            Dprint("Compilation successful with fallback Makefile\n")
            return 0

    Dprint("Compilation failed after 10 attempts + fallback\n")
    return 1

def compileTest(function): 
    # Functions will patch and compile the approximated function.
    path1 = "target"
    path2 = f"compilation_testing/ApxFiles_{function}"

    # for each key in approximated_functions_dict make a new folder

    copyFiles(path1, path2)

    apx_josn_file = f"apx_{function}.json"
    destination_file_path = os.path.join(path2, apx_josn_file)

    # Copy the file
    file_path = f"approximated_functions/apx_{function}.json"
    shutil.copy2(file_path, destination_file_path)

    lsp_patcher(path2,apx_josn_file)

    # Generate and test make file
    
    error_code = generateMakeFile(target_path=path2)

    if error_code == 1:
        log_path = os.path.join(path2, "compiler_log.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as file:
                return file.read()
        return "Makefile generation failed before compilation; no compiler log was produced."

    return ""
