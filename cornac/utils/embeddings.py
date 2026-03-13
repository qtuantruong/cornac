# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Utilities for loading and computing pre-trained embeddings.

Provides a high-level helper :func:`load_pretrained_embeddings` that
accepts raw data (text strings or PIL images) and a HuggingFace model
name, and returns a numpy feature matrix ready for use with any
:class:`~cornac.data.FeatureModality`.
"""

from typing import List, Optional, Union

import numpy as np

__all__ = ["load_pretrained_embeddings"]


def load_pretrained_embeddings(
    data: Union[List[str], list],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    modality: str = "text",
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute dense embeddings for a list of items using a pre-trained model.

    This is a convenience function that wraps common HuggingFace pipelines
    so that users can obtain feature matrices in a single call.

    Parameters
    ----------
    data : list of str **or** list of PIL.Image
        - For ``modality='text'``: a list of text strings.
        - For ``modality='image'``: a list of PIL images.
        - For ``modality='clip_text'`` / ``'clip_image'``: same as above but
          encoded with CLIP.

    model_name : str
        HuggingFace model identifier.  Sensible defaults:

        - **Text** (sentence-transformers): ``'sentence-transformers/all-MiniLM-L6-v2'``
        - **Text** (BERT): ``'bert-base-uncased'``
        - **Image** (CLIP): ``'openai/clip-vit-base-patch32'``

    modality : ``{'text', 'image', 'clip_text', 'clip_image'}``
        Type of encoding pipeline to use.

    batch_size : int, default 64
        Batch size for model inference.

    device : str or None
        PyTorch device.  If *None*, the library default is used.

    normalize : bool, default True
        L2-normalize the output embeddings.

    Returns
    -------
    embeddings : numpy.ndarray, shape ``(len(data), D)``
        Feature matrix where each row is a dense embedding.

    Examples
    --------
    >>> from cornac.utils.embeddings import load_pretrained_embeddings
    >>> texts = ["A great movie", "Terrible product"]
    >>> features = load_pretrained_embeddings(texts, modality="text")
    >>> features.shape
    (2, 384)
    """
    modality = modality.lower()

    if modality == "text":
        return _encode_sentences(data, model_name, batch_size, device, normalize)
    elif modality == "image":
        return _encode_images_vit(data, model_name, batch_size, device, normalize)
    elif modality in ("clip_text", "clip_image"):
        return _encode_clip(data, model_name, modality, batch_size, device, normalize)
    else:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            "Choose from: 'text', 'image', 'clip_text', 'clip_image'."
        )


# =========================================================================
# Private encoding backends
# =========================================================================


def _encode_sentences(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    normalize: bool,
) -> np.ndarray:
    """Encode text via ``sentence-transformers``."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings


def _encode_images_vit(
    images,
    model_name: str,
    batch_size: int,
    device: Optional[str],
    normalize: bool,
) -> np.ndarray:
    """Encode images via a HuggingFace ViT feature extractor."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    all_embs: list = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]
        inputs = extractor(images=batch, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token output
        embs = outputs.last_hidden_state[:, 0, :]
        if normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def _encode_clip(
    data,
    model_name: str,
    modality: str,
    batch_size: int,
    device: Optional[str],
    normalize: bool,
) -> np.ndarray:
    """Encode text or images using CLIP."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    all_embs: list = []
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]

        if modality == "clip_text":
            inputs = processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(dev)
            with torch.no_grad():
                embs = model.get_text_features(**inputs)
        else:
            batch = [
                img.convert("RGB") if img.mode != "RGB" else img for img in batch
            ]
            inputs = processor(images=batch, return_tensors="pt").to(dev)
            with torch.no_grad():
                embs = model.get_image_features(**inputs)

        if normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)
