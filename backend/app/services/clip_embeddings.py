"""CLIP image/text embedding for video frames."""

import open_clip
import torch
from PIL import Image

from app.config import settings

_model = None
_preprocess = None
_tokenizer = None
_device = "cpu"


def _load_clip():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval()
        _model.to(_device)
    return _model, _preprocess, _tokenizer


def embed_image(image_path: str) -> list[float]:
    model, preprocess, _ = _load_clip()
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()


def embed_text_clip(text: str) -> list[float]:
    model, _, tokenizer = _load_clip()
    tokens = tokenizer([text]).to(_device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()
