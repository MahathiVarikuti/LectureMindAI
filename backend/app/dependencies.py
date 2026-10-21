from app.db.chroma import collection
from app.services.retrieval_service import Retriever

# One retriever for the app; upload/delete routes call retriever.invalidate().
retriever = Retriever(collection)
