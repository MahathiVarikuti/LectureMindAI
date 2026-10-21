from functools import lru_cache

from app import config


@lru_cache(maxsize=None)
def get_model(name=None):
    # Imported lazily so the API (and its tests) start without loading torch.
    from sentence_transformers import SentenceTransformer

    from app.errors import AppError

    try:
        return SentenceTransformer(name or config.EMBEDDING_MODEL)
    except Exception as exc:  # e.g. no network on first run, so the model cannot be downloaded
        raise AppError(503, f"The embedding model could not be loaded: {exc}") from exc


def _query_prefix(name):
    # Only BGE-style models were trained with a query instruction.
    return config.QUERY_INSTRUCTION if "bge" in name.lower() else ""


def embed_documents(texts, model_name=None):
    name = model_name or config.EMBEDDING_MODEL
    return get_model(name).encode(
        list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )


def embed_query(query, model_name=None):
    name = model_name or config.EMBEDDING_MODEL
    return get_model(name).encode(_query_prefix(name) + query, normalize_embeddings=True)


# Backwards-compatible name used by older code.
generate_embedding = embed_documents
