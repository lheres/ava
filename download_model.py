import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
# The model name should match the one in your turing_ai.py script.
MODEL_NAME = "microsoft/GODEL-v1_1-base-seq2seq"
# ---------------------

# The cache directory is set by the ENV variable in the Dockerfile
cache_dir = os.getenv("TRANSFORMERS_CACHE")

print(f"Downloading and caching model: {MODEL_NAME}")

if cache_dir:
    # Download and cache the tokenizer and model to the specified directory
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
else:
    print("Warning: TRANSFORMERS_CACHE environment variable not set.")
    # Fallback to default behavior if the env var isn't set
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model and tokenizer have been successfully cached.")
