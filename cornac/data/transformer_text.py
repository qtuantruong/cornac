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

"""Transformer-based text modality for encoding text using pre-trained models.

This module provides a reusable :class:`TransformerTextModality` that wraps
HuggingFace `SentenceTransformer` models to encode text corpora into dense
embeddings.  It can be used by any Cornac model that needs textual item or
user representations (e.g. DMRL, FMRec, CLIPRec).
"""

import os
from typing import List, Optional

import numpy as np

from .modality import FeatureModality

__all__ = ["TransformerTextModality"]


class TransformerTextModality(FeatureModality):
    """Text modality backed by a HuggingFace SentenceTransformer encoder.

    Parameters
    ----------
    corpus : list of str, optional
        Raw text strings aligned with ``ids``.

    ids : list, optional
        User or item IDs aligned with ``corpus``.

    model_name_or_path : str, default ``'paraphrase-MiniLM-L6-v2'``
        Any model identifier accepted by ``sentence_transformers.SentenceTransformer``.

    preencode : bool, default False
        If *True*, the entire corpus is encoded at construction time and
        the resulting tensor is cached to ``cache_dir``.

    cache_dir : str, default ``'cornac_cache/encoded_text'``
        Directory used for persisting pre-encoded embeddings to avoid
        redundant computation across runs.

    batch_size : int, default 64
        Batch size used when encoding the corpus.

    device : str or None, default None
        Device string for the underlying SentenceTransformer model
        (e.g. ``'cuda:0'``).  If *None*, the library default is used.

    Attributes
    ----------
    output_dim : int
        Dimensionality of the output embeddings.

    preencoded : bool
        Whether the corpus has been pre-encoded.
    """

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        ids: Optional[List] = None,
        model_name_or_path: str = "paraphrase-MiniLM-L6-v2",
        preencode: bool = False,
        cache_dir: str = "cornac_cache/encoded_text",
        batch_size: int = 64,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(ids=ids, **kwargs)
        self.corpus = corpus
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._device = device
        self.preencoded = False

        # Lazy-load the model only when needed
        self._model = None
        self._output_dim = None

        if preencode and corpus is not None:
            self.preencode_entire_corpus()

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------
    @property
    def model(self):
        """Lazy-loaded SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name_or_path, device=self._device
            )
        return self._model

    @property
    def output_dim(self) -> int:
        """Dimensionality of the text embedding vectors."""
        if self._output_dim is None:
            self._output_dim = self.model[-1].pooling_output_dimension
        return self._output_dim

    # ------------------------------------------------------------------
    # Pre-encoding
    # ------------------------------------------------------------------
    def preencode_entire_corpus(self):
        """Encode the full corpus and cache to disk.

        If a cached file with matching IDs already exists it will be loaded
        instead of re-computing embeddings.
        """
        import torch

        path = os.path.join(self.cache_dir, "encoded_corpus.pt")
        id_path = os.path.join(self.cache_dir, "encoded_corpus_ids.pt")

        if os.path.exists(path) and os.path.exists(id_path):
            try:
                saved_ids = torch.load(id_path, weights_only=False)
                if saved_ids == self.ids:
                    self.features = torch.load(path, weights_only=False)
                    self.preencoded = True
                    return
            except Exception:
                pass  # cache miss – re-encode below

        if self.corpus is None:
            raise ValueError("Cannot pre-encode: corpus is None.")

        print(
            f"[TransformerTextModality] Pre-encoding {len(self.corpus)} texts "
            f"with '{self.model_name_or_path}'…"
        )
        self.features = self.model.encode(
            self.corpus,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )
        self.preencoded = True

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(self.features, path)
        torch.save(self.ids, id_path)

    # ------------------------------------------------------------------
    # On-the-fly encoding
    # ------------------------------------------------------------------
    def batch_encode(self, ids: List[int]):
        """Encode a batch of items on-the-fly by their indices.

        Parameters
        ----------
        ids : list of int
            Indices into ``self.corpus``.

        Returns
        -------
        encoded : torch.Tensor
            Shape ``(len(ids), output_dim)``.
        """
        if self.preencoded and self.features is not None:
            return self.features[ids]

        from operator import itemgetter

        texts = list(itemgetter(*ids)(self.corpus))
        return self.model.encode(texts, convert_to_tensor=True)

    # ------------------------------------------------------------------
    # Numpy helpers (for non-PyTorch models)
    # ------------------------------------------------------------------
    def batch_feature(self, batch_ids):
        """Return embeddings as a numpy array for given ``batch_ids``."""
        import torch

        feats = self.batch_encode(batch_ids)
        if isinstance(feats, torch.Tensor):
            return feats.cpu().numpy()
        return np.asarray(feats)
