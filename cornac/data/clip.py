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

"""CLIP-based multimodal modality for joint text and image encoding.

This module provides :class:`CLIPModality` that wraps a HuggingFace CLIP model
to produce aligned text **and** image embeddings in a shared latent space.
Useful for cross-modal retrieval (e.g. querying items by text description)
and cold-start recommendation where both text and image features are
available.
"""

import os
from typing import List, Optional

import numpy as np

from .modality import FeatureModality

__all__ = ["CLIPModality"]


class CLIPModality(FeatureModality):
    """Multi-modal modality that encodes text and/or images via CLIP.

    The resulting embeddings live in a **shared** vector space, meaning that
    text embeddings and image embeddings are directly comparable via cosine
    similarity.

    Parameters
    ----------
    corpus : list of str, optional
        Raw text strings aligned with ``ids``.

    images : list of PIL.Image, optional
        Loaded PIL images aligned with ``ids``.

    ids : list, optional
        User or item IDs.

    model_name : str, default ``'openai/clip-vit-base-patch32'``
        HuggingFace model identifier for the CLIP model.

    preencode : bool, default False
        Pre-encode all data at construction time.

    cache_dir : str, default ``'cornac_cache/encoded_clip'``
        Directory for caching pre-encoded features.

    batch_size : int, default 32
        Batch size for encoding.

    device : str or None, default None
        PyTorch device string.

    Attributes
    ----------
    text_features : numpy.ndarray or None
        Pre-encoded text feature matrix ``(N, D)``.

    image_features : numpy.ndarray or None
        Pre-encoded image feature matrix ``(N, D)``.

    features : numpy.ndarray or None
        Combined (averaged) feature matrix ``(N, D)`` when both text **and**
        image are available, otherwise whichever is present.

    output_dim : int
        Dimensionality of the CLIP embeddings.
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        images=None,
        ids: Optional[List] = None,
        model_name: str = "openai/clip-vit-base-patch32",
        preencode: bool = False,
        cache_dir: str = "cornac_cache/encoded_clip",
        batch_size: int = 32,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(ids=ids, **kwargs)
        self.corpus = corpus
        self.images = images
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None

        self.text_features = None
        self.image_features = None

        if preencode:
            self.preencode()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @property
    def output_dim(self) -> int:
        self._ensure_model()
        return self._model.config.projection_dim

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into CLIP embeddings.

        Returns
        -------
        embeddings : numpy.ndarray, shape ``(len(texts), output_dim)``
        """
        import torch

        self._ensure_model()
        all_embs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with torch.no_grad():
                embs = self._model.get_text_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    def encode_images(self, images) -> np.ndarray:
        """Encode a list of PIL images into CLIP embeddings.

        Returns
        -------
        embeddings : numpy.ndarray, shape ``(len(images), output_dim)``
        """
        import torch

        self._ensure_model()
        all_embs = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            # Ensure RGB
            batch = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]
            inputs = self._processor(
                images=batch, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                embs = self._model.get_image_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    # ------------------------------------------------------------------
    # Pre-encoding
    # ------------------------------------------------------------------
    def preencode(self):
        """Encode all available data (text, images, or both) and cache."""
        import torch

        text_path = os.path.join(self.cache_dir, "clip_text.npy")
        image_path = os.path.join(self.cache_dir, "clip_images.npy")
        id_path = os.path.join(self.cache_dir, "clip_ids.pt")

        # Try loading from cache
        cache_hit = False
        if os.path.exists(id_path):
            try:
                saved_ids = torch.load(id_path, weights_only=False)
                if saved_ids == self.ids:
                    if os.path.exists(text_path):
                        self.text_features = np.load(text_path)
                    if os.path.exists(image_path):
                        self.image_features = np.load(image_path)
                    cache_hit = True
            except Exception:
                pass

        if not cache_hit:
            os.makedirs(self.cache_dir, exist_ok=True)

            if self.corpus is not None:
                print(
                    f"[CLIPModality] Encoding {len(self.corpus)} texts "
                    f"with '{self.model_name}'…"
                )
                self.text_features = self.encode_texts(self.corpus)
                np.save(text_path, self.text_features)

            if self.images is not None:
                print(
                    f"[CLIPModality] Encoding {len(self.images)} images "
                    f"with '{self.model_name}'…"
                )
                self.image_features = self.encode_images(self.images)
                np.save(image_path, self.image_features)

            torch.save(self.ids, id_path)

        # Build the combined feature matrix
        self._build_combined_features()

    def _build_combined_features(self):
        """Combine text and image features into a single feature matrix."""
        if self.text_features is not None and self.image_features is not None:
            self.features = (self.text_features + self.image_features) / 2.0
        elif self.text_features is not None:
            self.features = self.text_features
        elif self.image_features is not None:
            self.features = self.image_features

    # ------------------------------------------------------------------
    # On-the-fly encoding
    # ------------------------------------------------------------------
    def batch_encode_text(self, ids: List[int]) -> np.ndarray:
        """Encode texts for the given indices."""
        if self.text_features is not None:
            return self.text_features[ids]
        from operator import itemgetter

        texts = list(itemgetter(*ids)(self.corpus))
        return self.encode_texts(texts)

    def batch_encode_image(self, ids: List[int]) -> np.ndarray:
        """Encode images for the given indices."""
        if self.image_features is not None:
            return self.image_features[ids]
        batch_images = [self.images[i] for i in ids]
        return self.encode_images(batch_images)

    def batch_feature(self, batch_ids):
        """Return combined (or available) embeddings for ``batch_ids``."""
        if self.features is not None:
            return self.features[batch_ids]

        text_feats = (
            self.batch_encode_text(batch_ids) if self.corpus is not None else None
        )
        image_feats = (
            self.batch_encode_image(batch_ids) if self.images is not None else None
        )

        if text_feats is not None and image_feats is not None:
            return (text_feats + image_feats) / 2.0
        return text_feats if text_feats is not None else image_feats
