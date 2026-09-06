import os

# 0 = No Summary; 1 = At very begining
APPROXIMATION_CONTEXT = 1
GIVE_FORMAT_EXAMPLES = 0
GIVE_LOOP_PERF_EXMAPLES = 1
GIVE_NEW_TECHNIQUE_EXAMPLES = 1
PRINT_LLM_CONVO = 0
ERROR_BOUND = 0.30
REVERSED_PDG = 0
TEXT_PLANING = 1
# Supported providers: OpenAI, Anthropic, HuggingFace, and Ollama.
# CHAI_API_NAME in .env overrides the default without requiring a source edit.
API_NAME = os.getenv("CHAI_API_NAME", "OpenAI")
DEBUG = 0
