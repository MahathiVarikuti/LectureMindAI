"""Central configuration. Every value can be overridden with an environment variable."""
import os


def _env(name, default, cast=str):
    value = os.getenv(name)
    return default if value is None else cast(value)


# --- Models -----------------------------------------------------------------
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE models expect this prefix on queries (not on documents).
QUERY_INSTRUCTION = _env(
    "QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
)
RERANKER_MODEL = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NLI_MODEL = _env("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:1.5b")
# Ollama's default context window silently truncates long prompts (it drops the
# *start* of the prompt, i.e. the grounding instructions).
LLM_NUM_CTX = _env("LLM_NUM_CTX", 8192, int)
LLM_TEMPERATURE = _env("LLM_TEMPERATURE", 0.0, float)

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = _env("CHUNK_SIZE", 800, int)  # characters
CHUNK_OVERLAP = _env("CHUNK_OVERLAP", 150, int)
MIN_CHUNK_CHARS = _env("MIN_CHUNK_CHARS", 40, int)

# --- Retrieval --------------------------------------------------------------
COLLECTION_NAME = _env("COLLECTION_NAME", "lecture_notes_v2")
CANDIDATE_K = _env("CANDIDATE_K", 20, int)  # per retriever, before fusion
TOP_K = _env("TOP_K", 4, int)  # chunks sent to the LLM
USE_HYBRID = _env("USE_HYBRID", "1") == "1"
USE_RERANKER = _env("USE_RERANKER", "1") == "1"
RRF_K = 60

# --- Hallucination control --------------------------------------------------
# Reranker relevance (0-1) the best chunk must reach, otherwise we abstain
# without calling the LLM. Calibrated on SQuAD v2 in/out-of-scope questions:
# python -m eval.run threshold --dataset squad_v2
# (0.125 maximised balanced accuracy at 0.967, keeping in-scope answer rate >= 0.95)
RELEVANCE_THRESHOLD = _env("RELEVANCE_THRESHOLD", 0.125, float)
# Chunks scoring below this fraction of the best chunk are dropped as noise.
RELATIVE_SCORE_CUTOFF = _env("RELATIVE_SCORE_CUTOFF", 0.05, float)
# Answer sentences whose NLI entailment prob is below this are "unsupported".
SUPPORT_THRESHOLD = _env("SUPPORT_THRESHOLD", 0.5, float)
# If fewer than this fraction of answer sentences are supported, flag the answer.
GROUNDEDNESS_THRESHOLD = _env("GROUNDEDNESS_THRESHOLD", 0.6, float)
USE_GROUNDING_CHECK = _env("USE_GROUNDING_CHECK", "1") == "1"

NOT_FOUND_ANSWER = "I could not find that information in the uploaded material."

# --- Chat memory ------------------------------------------------------------
MAX_HISTORY_MESSAGES = 10
