"""Fallback C-function discovery when clangd returns an incomplete symbol set."""

import os
import re


def _function_definition_pattern(function_name):
    """Return an anchored matcher that cannot consume comments/directives above it."""
    if not isinstance(function_name, str) or not re.fullmatch(r"[A-Za-z_]\w*", function_name):
        return None
    return re.compile(
        rf"(?m)^[ \t]*(?P<signature>"
        rf"(?:[A-Za-z_]\w*[ \t*]+)+"
        rf"{re.escape(function_name)}\s*\([^;{{}}]*\))"
        rf"(?:(?:[ \t\r\n]+)|(?:/\*.*?\*/)|(?://[^\n]*(?:\n|$)))*\{{",
        re.DOTALL | re.MULTILINE,
    )


def find_source_function_signature(source, function_name):
    """Return only a function's declaration header, excluding prior text."""
    pattern = _function_definition_pattern(function_name)
    match = pattern.search(source) if pattern else None
    return match.group("signature").strip() if match else None


def _matching_brace(source, opening_index):
    depth = 0
    state = "code"
    index = opening_index
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == '"':
                state = "string"
            elif char == "'":
                state = "char"
            elif char == "/" and nxt == "/":
                state = "line_comment"
                index += 1
            elif char == "/" and nxt == "*":
                state = "block_comment"
                index += 1
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        elif state in {"string", "char"}:
            if char == "\\":
                index += 1
            elif (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
        elif state == "line_comment" and char == "\n":
            state = "code"
        elif state == "block_comment" and char == "*" and nxt == "/":
            state = "code"
            index += 1
        index += 1
    return None


def find_source_function_span(source, function_name):
    """Return ``(start_char, end_char)`` for a C function definition."""
    pattern = _function_definition_pattern(function_name)
    match = pattern.search(source) if pattern else None
    if not match:
        return None
    opening = source.find("{", match.start(), match.end())
    closing = _matching_brace(source, opening)
    return (match.start(), closing + 1) if closing is not None else None


def find_source_function_line_range(source, function_name):
    span = find_source_function_span(source, function_name)
    if not span:
        return None
    start, end = span
    return source.count("\n", 0, start), source.count("\n", 0, end)


def recover_source_entities(function_names, target_files, base_dir="target"):
    """Recover definitions missing from LSP entity extraction."""
    recovered = []
    unresolved = set(function_names)
    for file_name in target_files:
        path = os.path.join(base_dir, file_name)
        try:
            with open(path, encoding="utf-8") as source_file:
                source = source_file.read()
        except OSError:
            continue
        for function_name in list(unresolved):
            span = find_source_function_span(source, function_name)
            if not span:
                continue
            start, end = span
            recovered.append({
                "filePath": os.path.abspath(path),
                "functionName": function_name,
                "completeFunction": source[start:end].strip(),
            })
            unresolved.remove(function_name)
    return recovered

