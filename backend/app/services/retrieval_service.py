"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with Reciprocal Rank
Fusion, then re-scored by a cross-encoder reranker.

Why each stage:
  * dense vectors catch paraphrases ("cost function" ~ "loss")
  * BM25 catches exact terms embeddings blur: acronyms, symbols, names, formulas
  * the cross-encoder reads query + chunk together, which is far more accurate
    than comparing two independent vectors, and it gives a calibrated 0-1
    relevance score we can threshold on to refuse out-of-scope questions.
"""
import math
import re
import threading
from functools import lru_cache

from rank_bm25 import BM25Okapi
from app import config
from app.db.chroma import build_where
from app.services.embedding_service import embed_query

_STOPWORDS = set(
    "a an the of to in on for and or is are was were be been being it its this that "
    "these those with as by at from what which who whom how why when where do does did "
    "can could should would will shall may might i you he she we they me him her us them "
    "my your our their about into than then there here so not no".split()
)


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=None)
def get_reranker(name=None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name or config.RERANKER_MODEL)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def rerank_scores(query, texts, model_name=None):
    if not texts:
        return []
    model = get_reranker(model_name)
    raw = model.predict([(query, t) for t in texts], show_progress_bar=False)
    scores = [float(s) for s in raw]
    # ms-marco cross-encoders output logits; map to 0-1 so thresholds are portable.
    if any(s < 0 or s > 1 for s in scores):
        scores = [_sigmoid(s) for s in scores]
    return scores


class Retriever:
    def __init__(
        self,
        collection,
        embedding_model=None,
        use_hybrid=None,
        use_reranker=None,
        candidate_k=None,
        reranker_model=None,
    ):
        self.collection = collection
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.use_hybrid = config.USE_HYBRID if use_hybrid is None else use_hybrid
        self.use_reranker = config.USE_RERANKER if use_reranker is None else use_reranker
        self.candidate_k = candidate_k or config.CANDIDATE_K
        self.reranker_model = reranker_model
        # Score scales differ: reranker probability vs. raw cosine similarity.
        self.relevance_threshold = (
            config.RELEVANCE_THRESHOLD if config.RELEVANCE_THRESHOLD is not None
            else config.RERANKER_RELEVANCE_THRESHOLD if self.use_reranker
            else config.DENSE_RELEVANCE_THRESHOLD
        )
        self._bm25 = None
        self._lock = threading.Lock()

    def invalidate(self):
        """Call after adding/deleting chunks so the BM25 index is rebuilt."""
        with self._lock:
            self._bm25 = None

    def _bm25_index(self):
        with self._lock:
            if self._bm25 is None:
                data = self.collection.get(include=["documents", "metadatas"])
                ids, docs = data["ids"], data["documents"] or []
                metas = data["metadatas"] or [{}] * len(ids)
                tokens = [tokenize(d or "") for d in docs]
                index = BM25Okapi(tokens) if ids else None
                self._bm25 = (index, ids, docs, metas)
            return self._bm25

    def _dense(self, query, document_ids, session_id=None):
        count = self.collection.count()
        if count == 0:
            return []
        params = {
            "query_embeddings": [embed_query(query, self.embedding_model).tolist()],
            "n_results": min(self.candidate_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        where = build_where(session_id, document_ids)
        if where:
            params["where"] = where
        res = self.collection.query(**params)
        hits = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            if doc:
                hits.append({"id": cid, "text": doc, "metadata": meta or {}, "dense_score": 1 - dist})
        return hits

    def _sparse(self, query, document_ids, session_id=None):
        index, ids, docs, metas = self._bm25_index()
        q = tokenize(query)
        if index is None or not q:
            return []
        scores = index.get_scores(q)
        allowed = set(document_ids) if document_ids else None
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= self.candidate_k:
                break
            meta = metas[i] or {}
            if allowed is not None and meta.get("document_id") not in allowed:
                continue
            # Never surface another session's chunks, even though the BM25 index spans all.
            if session_id and meta.get("session_id") != session_id:
                continue
            hits.append({"id": ids[i], "text": docs[i], "metadata": meta, "bm25_score": float(scores[i])})
        return hits

    def retrieve(self, query, document_ids=None, top_k=None, session_id=None):
        """Return candidate chunks sorted best-first. Each has a `score` in 0-1:
        the reranker probability if reranking is on, else the cosine similarity."""
        top_k = top_k or self.candidate_k
        dense = self._dense(query, document_ids, session_id)
        sparse = self._sparse(query, document_ids, session_id) if self.use_hybrid else []

        fused = {}
        for ranked in (dense, sparse):
            for rank, hit in enumerate(ranked):
                entry = fused.setdefault(hit["id"], {**hit, "rrf": 0.0})
                entry.update({k: v for k, v in hit.items() if k.endswith("_score")})
                entry["rrf"] += 1.0 / (config.RRF_K + rank + 1)

        candidates = sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)
        if not candidates:
            return []

        if self.use_reranker:
            scores = rerank_scores(query, [c["text"] for c in candidates], self.reranker_model)
            for c, s in zip(candidates, scores):
                c["score"] = s
            candidates.sort(key=lambda c: c["score"], reverse=True)
        else:
            for c in candidates:
                c["score"] = c.get("dense_score", 0.0)
            if not self.use_hybrid:
                candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:top_k]


def select_context(candidates, top_k=None, threshold=None, relative_cutoff=None):
    """Apply the relevance threshold. Returns (kept_chunks, abstain_reason)."""
    top_k = top_k or config.TOP_K
    if threshold is None:
        threshold = config.RELEVANCE_THRESHOLD
        if threshold is None:
            threshold = config.DENSE_RELEVANCE_THRESHOLD
    relative_cutoff = config.RELATIVE_SCORE_CUTOFF if relative_cutoff is None else relative_cutoff
    if not candidates:
        return [], "no_documents"
    best = max(c["score"] for c in candidates)
    if best < threshold:
        return [], "low_relevance"
    kept = [c for c in candidates[:top_k] if c["score"] >= max(threshold, best * relative_cutoff)]
    return kept, None
