import os

MODEL_NAME = os.environ.get("MODEL_NAME", "sberbank-ai/rugpt3small_based_on_gpt2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 100))
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", 2))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", 1.1))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
TOP_P = float(os.environ.get("TOP_P", 0.95))
TOP_K = int(os.environ.get("TOP_K", 50))
