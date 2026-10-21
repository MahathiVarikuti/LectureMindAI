import os

import chromadb

from app import config

CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Cosine space: embeddings are L2-normalised, so distance = 1 - cosine similarity.
# (The old "lecture_notes" collection used L2 + MiniLM embeddings and is not
# compatible with the new embedding model; re-upload documents after upgrading.)
collection = client.get_or_create_collection(
    name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
