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

"""CLIP-based Cross-Modal Recommender (CLIPRec).

Leverages OpenAI CLIP embeddings for **cross-modal** item recommendation.
Items can be represented by text, images, or both.  The model supports
zero-shot recommendation for cold-start items: because CLIP produces
aligned text-image embeddings, new items with only a text description
(or only an image) can still be scored against existing user profiles.

This model corresponds to the **Feature-Based** paradigm in the
FM4RecSys survey, with CLIP as the foundation model.
"""

import numpy as np
from tqdm.auto import tqdm

from ..recommender import Recommender, ANNMixin, MEASURE_COSINE
from ...exception import CornacException
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform


class CLIPRec(Recommender, ANNMixin):
    """CLIP-based Cross-Modal Recommender.

    Parameters
    ----------
    name : str, default ``'CLIPRec'``
    k : int, default 64
        User latent factor dimension.
    n_epochs : int, default 15
    batch_size : int, default 256
    learning_rate : float, default 1e-3
    lambda_reg : float, default 1e-4
    use_gpu : bool, default True
    trainable : bool, default True
    verbose : bool, default True
    seed : int or None
    """

    def __init__(
        self,
        name="CLIPRec",
        k=64,
        n_epochs=15,
        batch_size=256,
        learning_rate=1e-3,
        lambda_reg=1e-4,
        use_gpu=True,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.use_gpu = use_gpu
        self.seed = seed

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, train_set, val_set=None):
        """Fit CLIPRec.

        The training set must carry a ``CLIPModality`` (or any
        :class:`~cornac.data.FeatureModality` with CLIP features)
        attached via ``item_embedding`` or ``item_text`` / ``item_image``.
        """
        Recommender.fit(self, train_set, val_set)

        # Obtain CLIP features
        clip_features = self._get_clip_features(train_set)
        if clip_features is None:
            raise CornacException(
                "CLIPRec requires CLIP embeddings in item_embedding, "
                "item_text, or item_image modality."
            )
        clip_features = clip_features[: self.total_items].astype(np.float32)
        clip_dim = clip_features.shape[1]

        # Normalise CLIP features (they should already be, but be safe)
        norms = np.linalg.norm(clip_features, axis=1, keepdims=True) + 1e-8
        clip_features = clip_features / norms

        self.item_clip = clip_features

        # User factors project into the same CLIP-dim space
        rng = get_rng(self.seed)
        self.user_factors = xavier_uniform((self.total_users, clip_dim), rng)

        if self.trainable:
            self._fit_bpr(train_set, clip_features)

        return self

    def _get_clip_features(self, train_set):
        """Extract CLIP feature matrix from train_set modalities."""
        for attr in ("item_embedding", "item_text", "item_image"):
            mod = getattr(train_set, attr, None)
            if mod is not None and hasattr(mod, "features") and mod.features is not None:
                f = mod.features
                if hasattr(f, "cpu"):
                    f = f.cpu().numpy()
                return np.asarray(f, dtype=np.float32)
        return None

    # ------------------------------------------------------------------
    # BPR training
    # ------------------------------------------------------------------
    def _fit_bpr(self, train_set, clip_features):
        import torch

        device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        C = torch.tensor(clip_features, device=device)
        U = torch.tensor(self.user_factors, device=device, requires_grad=True)

        opt = torch.optim.Adam([U], lr=self.learning_rate)

        for epoch in range(1, self.n_epochs + 1):
            total_loss = 0.0
            count = 0
            progress = tqdm(
                total=train_set.num_batches(self.batch_size),
                desc=f"Epoch {epoch}/{self.n_epochs}",
                disable=not self.verbose,
            )
            for u, i, j in train_set.uij_iter(self.batch_size, shuffle=True):
                x_uij = (U[u] * (C[i] - C[j])).sum(dim=1)
                loss = -torch.nn.functional.logsigmoid(x_uij).sum()
                loss += self.lambda_reg * U[u].pow(2).sum()

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                count += len(u)
                if count % (self.batch_size * 10) == 0:
                    progress.set_postfix(loss=total_loss / count)
                progress.update(1)
            progress.close()

        self.user_factors = U.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            return self.user_factors[user_idx] @ self.item_clip.T
        else:
            return self.user_factors[user_idx] @ self.item_clip[item_idx]

    # ------------------------------------------------------------------
    # Zero-shot helpers
    # ------------------------------------------------------------------
    def score_text_query(self, query_text, clip_modality=None, top_k=10):
        """Score all items against a free-text query using CLIP.

        Parameters
        ----------
        query_text : str
            Natural language description of what to recommend.
        clip_modality : CLIPModality, optional
            A CLIPModality with a loaded model.  If *None* one will be
            created with default model.
        top_k : int
            Number of top items to return.

        Returns
        -------
        ranked_items : list of (item_id, score)
        """
        if clip_modality is None:
            from ...data.clip import CLIPModality

            clip_modality = CLIPModality()

        text_emb = clip_modality.encode_texts([query_text])
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        scores = text_emb @ self.item_clip.T
        scores = scores.flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.item_ids[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # ANN support
    # ------------------------------------------------------------------
    def get_vector_measure(self):
        return MEASURE_COSINE

    def get_user_vectors(self):
        return self.user_factors

    def get_item_vectors(self):
        return self.item_clip
