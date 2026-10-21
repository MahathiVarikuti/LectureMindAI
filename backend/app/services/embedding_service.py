from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=None)
def get_model(name=None):
    return SentenceTransformer(name or config.EMBEDDING_MODEL)


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
