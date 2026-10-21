"""CLIP embeddings for video frames (images) and visual queries (text).

CLIP maps images and text into one shared space, but that space is NOT shared with the
SentenceTransformers text embeddings, which is why frames live in their own collection.
Model: openai/clip-vit-base-patch32 (the smallest CLIP, chosen for CPU speed).
"""
from functools import lru_cache

from app.config import settings

BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load():
    # Heavy imports are deferred so the API starts quickly and tests need no torch.
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from app.errors import AppError

    try:
        model = CLIPModel.from_pretrained(settings.clip_model).eval()
        processor = CLIPProcessor.from_pretrained(settings.clip_model)
    except Exception as exc:
        raise AppError(503, f"The CLIP model could not be loaded: {exc}") from exc
    return torch, model, processor


def _as_tensor(features):
    # Newer transformers versions may wrap the projection in an output object.
    return features if hasattr(features, "norm") else features.pooler_output


def _normalise(torch, features):
    features = _as_tensor(features)
    return (features / features.norm(dim=-1, keepdim=True)).cpu().tolist()


def embed_images(paths):
    """Embed image files in small batches (keeps CPU memory bounded)."""
    from PIL import Image

    torch, model, processor = _load()
    out = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for p in paths[start:start + BATCH_SIZE]:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            out.extend(_normalise(torch, model.get_image_features(**inputs)))
    return out


def embed_text(text):
    torch, model, processor = _load()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        return _normalise(torch, model.get_text_features(**inputs))[0]
