"""Post-generation hallucination check.

Each sentence of the answer is treated as a hypothesis and every retrieved
chunk as a premise. A small NLI (natural language inference) model estimates
P(chunk entails sentence). A sentence is "supported" if at least one chunk
entails it. Groundedness = fraction of supported sentences.
"""
import re
from functools import lru_cache

import numpy as np
from app import config

_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "with",
}


@lru_cache(maxsize=None)
def get_nli_model(name=None):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name or config.NLI_MODEL)
    labels = {v.lower(): int(k) for k, v in model.model.config.id2label.items()}
    return model, labels["entailment"]


def split_sentences(text):
    text = _CITATION.sub("", text)
    lines = [ln.strip(" -*\t") for ln in text.splitlines()]
    sentences = []
    for line in lines:
        if not line:
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line) if s.strip())
    return sentences


def _is_claim(sentence):
    # Headings, list labels and "Here is a summary:" style lines carry no facts.
    return len(sentence.split()) >= 4 and not sentence.endswith(":")


def _lexical_support(sentence, contexts):
    """Catch near-verbatim extracted bullets that strict NLI can underrate."""
    claim = {
        token for token in _WORD.findall(sentence.lower())
        if token not in _STOPWORDS
    }
    if len(claim) < 5:
        return False
    return any(
        len(claim & {
            token for token in _WORD.findall(context.lower())
            if token not in _STOPWORDS
        }) / len(claim) >= 0.72
        for context in contexts
    )


def entailment_scores(premises, hypotheses, model_name=None):
    """Matrix [len(hypotheses), len(premises)] of entailment probabilities."""
    model, ent_idx = get_nli_model(model_name)
    pairs = [(p, h) for h in hypotheses for p in premises]
    logits = np.asarray(model.predict(pairs, show_progress_bar=False, batch_size=16))
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs[:, ent_idx].reshape(len(hypotheses), len(premises))


def check_grounding(answer, contexts, support_threshold=None, model_name=None):
    support_threshold = config.SUPPORT_THRESHOLD if support_threshold is None else support_threshold
    claims = [s for s in split_sentences(answer) if _is_claim(s)]
    if not claims or not contexts:
        return {"groundedness": 1.0 if not claims else 0.0, "sentences": [], "unsupported": []}

    support = entailment_scores(contexts, claims, model_name).max(axis=1)
    sentences = []
    for sentence, probability in zip(claims, support):
        lexical = _lexical_support(sentence, contexts)
        sentences.append(
            {
                "text": sentence,
                "support": round(float(max(probability, 1.0 if lexical else 0.0)), 3),
                "supported": bool(probability >= support_threshold or lexical),
            }
        )
    supported = sum(s["supported"] for s in sentences)
    return {
        "groundedness": round(supported / len(sentences), 3),
        "sentences": sentences,
        "unsupported": [s["text"] for s in sentences if not s["supported"]],
    }
