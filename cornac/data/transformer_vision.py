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

"""Transformer-based vision modality for encoding images with pre-trained ViT.

This module provides :class:`TransformerVisionModality` that wraps a
``torchvision`` Vision Transformer (ViT) model for image encoding.  It can be
used by any Cornac model that needs visual item or user representations.
"""

import os
from typing import List, Optional

import numpy as np

from .modality import FeatureModality

__all__ = ["TransformerVisionModality"]


class TransformerVisionModality(FeatureModality):
    """Image modality backed by a ``torchvision`` ViT encoder.

    Parameters
    ----------
    images : list of PIL.Image, optional
        Loaded PIL images aligned with ``ids``.

    ids : list, optional
        User or item IDs aligned with ``images``.

    model_weights : torchvision WeightsEnum, optional
        Pre-trained weights for the ViT model.  Defaults to
        ``models.ViT_B_16_Weights.DEFAULT`` (a smaller, more practical default
        than the ViT-H/14 that was previously hard-coded in DMRL).

    preencode : bool, default False
        If *True*, all images are encoded at construction time.

    cache_dir : str, default ``'cornac_cache/encoded_images'``
        Directory for caching pre-encoded features.

    batch_size : int, default 32
        Batch size for encoding.

    device : str or None, default None
        PyTorch device string.  If *None*, CUDA is used when available.

    Attributes
    ----------
    output_dim : int
        Dimensionality of the output embeddings.

    preencoded : bool
        Whether the images have been pre-encoded.
    """

    def __init__(
        self,
        images=None,
        ids: Optional[List] = None,
        model_weights=None,
        preencode: bool = False,
        cache_dir: str = "cornac_cache/encoded_images",
        batch_size: int = 32,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(ids=ids, **kwargs)
        self.images = images
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.preencoded = False

        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-load model
        self._model = None
        self._model_weights = model_weights
        self._image_size = None
        self._output_dim = None
        self._transforms = None

        if preencode and images is not None:
            self.preencode_images()

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------
    def _ensure_model(self):
        """Load the ViT model and set up transforms if not already done."""
        if self._model is not None:
            return

        import torch
        from torchvision import models, transforms

        if self._model_weights is None:
            self._model_weights = models.ViT_B_16_Weights.DEFAULT

        model = models.vit_b_16(weights=self._model_weights)
        model.heads = torch.nn.Identity()  # remove classification head
        model.eval()
        model = model.to(self._device)
        self._model = model
        self._image_size = (model.image_size, model.image_size)

        # Use the weight's recommended transforms when available
        if hasattr(self._model_weights, "transforms"):
            self._transforms = self._model_weights.transforms()
        else:
            self._transforms = transforms.Compose(
                [
                    transforms.Resize(self._image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    @property
    def output_dim(self) -> int:
        """Dimensionality of the image embedding vectors."""
        if self._output_dim is None:
            self._ensure_model()
            # ViT-B/16 hidden dim = 768
            import torch

            dummy = torch.randn(1, 3, *self._image_size, device=self._device)
            with torch.no_grad():
                out = self._model(dummy)
            self._output_dim = out.shape[-1]
        return self._output_dim

    # ------------------------------------------------------------------
    # Image → tensor
    # ------------------------------------------------------------------
    def _transform_images(self, images) -> "torch.Tensor":
        """Convert a list of PIL images to a batched tensor."""
        import torch

        self._ensure_model()
        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self._transforms(img))
        return torch.stack(tensors).to(self._device)

    # ------------------------------------------------------------------
    # Pre-encoding
    # ------------------------------------------------------------------
    def preencode_images(self):
        """Encode all images and cache results to disk."""
        import torch

        path = os.path.join(self.cache_dir, "encoded_images.pt")
        id_path = os.path.join(self.cache_dir, "encoded_images_ids.pt")

        if os.path.exists(path) and os.path.exists(id_path):
            try:
                saved_ids = torch.load(id_path, weights_only=False)
                if saved_ids == self.ids:
                    self.features = torch.load(path, weights_only=False)
                    self.preencoded = True
                    return
            except Exception:
                pass

        if self.images is None:
            raise ValueError("Cannot pre-encode: images is None.")

        self._ensure_model()
        print(
            f"[TransformerVisionModality] Pre-encoding {len(self.images)} images…"
        )

        all_features = []
        for start in range(0, len(self.images), self.batch_size):
            batch = self.images[start : start + self.batch_size]
            tensor_batch = self._transform_images(batch)
            with torch.no_grad():
                encoded = self._model(tensor_batch)
            all_features.append(encoded.cpu())

        self.features = torch.cat(all_features, dim=0)
        self.preencoded = True

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(self.features, path)
        torch.save(self.ids, id_path)

    # ------------------------------------------------------------------
    # On-the-fly encoding
    # ------------------------------------------------------------------
    def batch_encode(self, ids: List[int]):
        """Encode a batch of images on-the-fly by their indices.

        Parameters
        ----------
        ids : list of int
            Indices into ``self.images``.

        Returns
        -------
        encoded : torch.Tensor
            Shape ``(len(ids), output_dim)``.
        """
        import torch

        if self.preencoded and self.features is not None:
            return self.features[ids]

        self._ensure_model()
        batch_images = [self.images[i] for i in ids]
        tensor_batch = self._transform_images(batch_images)
        with torch.no_grad():
            return self._model(tensor_batch)

    # ------------------------------------------------------------------
    # Numpy helpers
    # ------------------------------------------------------------------
    def batch_feature(self, batch_ids):
        """Return embeddings as a numpy array for given ``batch_ids``."""
        import torch

        feats = self.batch_encode(batch_ids)
        if isinstance(feats, torch.Tensor):
            return feats.cpu().numpy()
        return np.asarray(feats)
