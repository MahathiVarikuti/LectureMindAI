"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_file() -> str:
    """Find the project root .env whether the app is started from the repo root or backend dir."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path.cwd() / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_timeout: int = 120

    embedding_model: str = "all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"

    whisper_model: str = "base"
    whisper_timeout: int = 600

    chunk_size: int = 1000
    chunk_overlap: int = 200

    frame_sample_interval: float = 7.0

    upload_dir: str = "./data/uploads"
    chroma_dir: str = "./data/chroma"
    sqlite_path: str = "./data/chat_history.db"

    max_pdf_size_mb: int = 25
    max_media_duration_minutes: int = 30
    max_storage_per_session_mb: int = 500
    max_documents_per_session: int = 20

    rate_limit_query: int = 10
    rate_limit_upload: int = 5

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def sqlite_db_path(self) -> Path:
        return Path(self.sqlite_path)

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def max_media_duration_seconds(self) -> float:
        return self.max_media_duration_minutes * 60

    @property
    def max_storage_per_session_bytes(self) -> int:
        return self.max_storage_per_session_mb * 1024 * 1024


settings = Settings()
