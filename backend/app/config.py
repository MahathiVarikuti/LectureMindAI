"""Central configuration, loaded from the environment / a .env file via pydantic-settings.

Nothing is hardcoded elsewhere: URLs, model names, paths, limits and CORS origins
all come from here. See .env.example for the documented list of variables.

The module-level UPPERCASE names at the bottom mirror the settings so that the RAG
primitives and the evaluation harness (which read `config.X` at call time and may
override it) keep working unchanged.
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # --- Models -------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    # BGE models were trained with this instruction on *queries* only.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    clip_model: str = "openai/clip-vit-base-patch32"
    whisper_model: str = "base"  # tiny | base | small | medium (CPU: base is the sweet spot)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # --- Ollama -------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    llm_num_ctx: int = 8192
    llm_temperature: float = 0.0
    ollama_timeout: int = 120  # seconds; a hung Ollama must not hang the API
    whisper_timeout: int = 900  # seconds of wall clock for one transcription

    # --- Chunking -----------------------------------------------------------
    # Larger chunks carry more context per hit but dilute the embedding and cost more
    # LLM tokens; more overlap avoids cutting an idea in half at a boundary but
    # stores duplicate text. 1000/200 is a common balance for lecture prose.
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_chars: int = 40

    # --- Retrieval ----------------------------------------------------------
    text_collection: str = "text_chunks"
    image_collection: str = "image_frames"
    candidate_k: int = 20
    top_k: int = 4
    # V1 is dense-only by spec. These optional, off-by-default features come from the
    # evaluation work (see docs/evaluation.md) and can be enabled without code changes.
    use_hybrid: bool = False
    use_reranker: bool = False
    use_grounding_check: bool = False
    # Minimum score of the best chunk, otherwise we answer "not found" without calling the LLM.
    # Unset -> 0.125 when the reranker is on (reranker probability), else 0.15 (cosine, MiniLM).
    relevance_threshold: float | None = None
    relative_score_cutoff: float = 0.05
    image_min_score: float = 0.2  # raw CLIP text-image cosine; typical matches are 0.2-0.35
    max_image_hits: int = 3
    support_threshold: float = 0.5
    groundedness_threshold: float = 0.6

    # --- Video --------------------------------------------------------------
    frame_sample_interval: float = 7.0  # seconds between sampled frames

    # --- Storage ------------------------------------------------------------
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"
    sqlite_path: Path = BACKEND_DIR / "data" / "chat_history.db"

    # --- Resource limits (enforced before any expensive work) ---------------
    max_pdf_size_mb: int = 25
    max_media_size_mb: int = 500
    max_media_duration_minutes: int = 30
    max_storage_per_session_mb: int = 500
    max_documents_per_session: int = 20
    max_concurrent_ingestions: int = 1  # models are memory-hungry; serialize heavy work
    max_stored_messages_per_session: int = 500
    max_history_messages: int = 10  # sent to the LLM for query rewriting

    # --- Rate limiting (requests/minute per client IP) ----------------------
    rate_limit_query: int = 10
    rate_limit_upload: int = 5

    # --- CORS ---------------------------------------------------------------
    # Comma-separated origins. "*" is for local development only.
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def max_media_size_bytes(self) -> int:
        return self.max_media_size_mb * 1024 * 1024

    @property
    def max_media_duration_seconds(self) -> int:
        return self.max_media_duration_minutes * 60

    @property
    def max_storage_per_session_bytes(self) -> int:
        return self.max_storage_per_session_mb * 1024 * 1024


settings = Settings()

# --- Module-level aliases (see module docstring) ----------------------------
EMBEDDING_MODEL = settings.embedding_model
QUERY_INSTRUCTION = settings.query_instruction
RERANKER_MODEL = settings.reranker_model
NLI_MODEL = settings.nli_model
OLLAMA_URL = settings.ollama_url
LLM_MODEL = settings.llm_model
LLM_NUM_CTX = settings.llm_num_ctx
LLM_TEMPERATURE = settings.llm_temperature
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MIN_CHUNK_CHARS = settings.min_chunk_chars
COLLECTION_NAME = settings.text_collection
CANDIDATE_K = settings.candidate_k
TOP_K = settings.top_k
USE_HYBRID = settings.use_hybrid
USE_RERANKER = settings.use_reranker
USE_GROUNDING_CHECK = settings.use_grounding_check
RRF_K = 60
RELEVANCE_THRESHOLD = settings.relevance_threshold  # None -> chosen per retriever mode
RELATIVE_SCORE_CUTOFF = settings.relative_score_cutoff
SUPPORT_THRESHOLD = settings.support_threshold
GROUNDEDNESS_THRESHOLD = settings.groundedness_threshold
RERANKER_RELEVANCE_THRESHOLD = 0.125  # calibrated on SQuAD v2, see docs/evaluation.md
DENSE_RELEVANCE_THRESHOLD = 0.15  # cosine similarity for the default MiniLM embeddings (typical unrelated pairs score < 0.1)
MAX_HISTORY_MESSAGES = settings.max_history_messages

NOT_FOUND_ANSWER = "I could not find that information in the uploaded material."
