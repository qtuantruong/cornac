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

"""Feature-Based Foundation Model Recommender (FMRec).

A general-purpose recommender that uses pre-trained transformer embeddings
(BERT, SentenceTransformers, ViT, etc.) as item feature extractors and
combines them with collaborative filtering signals via a lightweight
projection layer and BPR or BCE loss.

This model is the Cornac counterpart to the **Feature-Based paradigm**
described in the FM4RecSys survey (arXiv 2504.16420).
"""

import numpy as np
from tqdm.auto import tqdm

from ..recommender import Recommender, ANNMixin, MEASURE_DOT
from ...exception import CornacException
from ...utils import get_rng
from ...utils.init_utils import zeros, xavier_uniform


class FMRec(Recommender, ANNMixin):
    """Feature-Based Foundation Model Recommender.

    Uses pre-trained foundation-model embeddings (text and/or image) as
    item side-information and learns user/item latent factors via BPR.

    Parameters
    ----------
    name : str, default ``'FMRec'``
    k : int, default 50
        Latent factor dimensionality for user/item collaborative embeddings.
    n_epochs : int, default 20
    batch_size : int, default 256
    learning_rate : float, default 1e-3
    lambda_reg : float, default 1e-4
        L2 regularisation weight.
    use_gpu : bool, default True
    projection_dim : int or None
        If set, project FM embeddings to this dimensionality before fusion.
        If *None*, raw FM dimension is used.
    trainable : bool, default True
    verbose : bool, default True
    seed : int or None
    """

    def __init__(
        self,
        name="FMRec",
        k=50,
        n_epochs=20,
        batch_size=256,
        learning_rate=1e-3,
        lambda_reg=1e-4,
        use_gpu=True,
        projection_dim=None,
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
        self.projection_dim = projection_dim
        self.seed = seed

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, train_set, val_set=None):
        """Fit FMRec to ``train_set``.

        At least one of ``item_text`` or ``item_image`` modalities must
        be present in ``train_set`` (as a
        :class:`~cornac.data.FeatureModality` with pre-computed features).
        """
        Recommender.fit(self, train_set, val_set)

        # Gather item features ------------------------------------------
        features = self._collect_features(train_set)
        if features is None:
            raise CornacException(
                "FMRec requires at least one of item_text, item_image, "
                "or item_embedding modalities with pre-computed features."
            )
        features = features[: self.total_items].astype(np.float32)
        feat_dim = features.shape[1]

        # Determine projection target dimension
        proj_dim = self.projection_dim or feat_dim

        # Init parameters ------------------------------------------------
        rng = get_rng(self.seed)
        self.user_factors = xavier_uniform((self.total_users, self.k + proj_dim), rng)
        self.item_factors = xavier_uniform((self.total_items, self.k), rng)
        self.item_bias = zeros(self.total_items)
        self.projection = xavier_uniform((feat_dim, proj_dim), rng)

        # Pre-compute projected features
        self.item_fm_proj = features @ self.projection  # (n_items, proj_dim)

        if self.trainable:
            self._fit_bpr(train_set, features)

        return self

    def _collect_features(self, train_set):
        """Combine all available FM feature matrices."""
        parts = []
        for attr in ("item_text", "item_image", "item_embedding"):
            mod = getattr(train_set, attr, None)
            if mod is not None and hasattr(mod, "features") and mod.features is not None:
                f = mod.features
                # Handle torch tensors
                if hasattr(f, "cpu"):
                    f = f.cpu().numpy()
                parts.append(np.asarray(f, dtype=np.float32))
        if not parts:
            return None
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # BPR training (PyTorch)
    # ------------------------------------------------------------------
    def _fit_bpr(self, train_set, raw_features):
        import torch

        device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        F = torch.tensor(raw_features, device=device)
        U = torch.tensor(self.user_factors, device=device, requires_grad=True)
        I = torch.tensor(self.item_factors, device=device, requires_grad=True)
        B = torch.tensor(self.item_bias, device=device, requires_grad=True)
        P = torch.tensor(self.projection, device=device, requires_grad=True)

        opt = torch.optim.Adam([U, I, B, P], lr=self.learning_rate)

        for epoch in range(1, self.n_epochs + 1):
            total_loss = 0.0
            count = 0
            progress = tqdm(
                total=train_set.num_batches(self.batch_size),
                desc=f"Epoch {epoch}/{self.n_epochs}",
                disable=not self.verbose,
            )
            for u, i, j in train_set.uij_iter(self.batch_size, shuffle=True):
                fp_i = F[i] @ P  # (bs, proj_dim)
                fp_j = F[j] @ P

                # Composite item representation: [CF factors | FM projection]
                item_i = torch.cat([I[i], fp_i], dim=1)
                item_j = torch.cat([I[j], fp_j], dim=1)

                x_uij = (
                    B[i] - B[j]
                    + (U[u] * (item_i - item_j)).sum(dim=1)
                )
                loss = -torch.nn.functional.logsigmoid(x_uij).sum()
                loss += self.lambda_reg * (
                    U[u].pow(2).sum() + I[i].pow(2).sum() + I[j].pow(2).sum()
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                count += len(u)
                if count % (self.batch_size * 10) == 0:
                    progress.set_postfix(loss=total_loss / count)
                progress.update(1)
            progress.close()

        # Persist learned parameters
        self.user_factors = U.detach().cpu().numpy()
        self.item_factors = I.detach().cpu().numpy()
        self.item_bias = B.detach().cpu().numpy()
        self.projection = P.detach().cpu().numpy()
        self.item_fm_proj = (F @ P).detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            # Composite item vectors
            item_vec = np.concatenate(
                [self.item_factors, self.item_fm_proj], axis=1
            )
            scores = self.item_bias.copy()
            scores += self.user_factors[user_idx] @ item_vec.T
            return scores
        else:
            item_vec = np.concatenate(
                [self.item_factors[item_idx], self.item_fm_proj[item_idx]]
            )
            return self.item_bias[item_idx] + self.user_factors[user_idx] @ item_vec

    # ------------------------------------------------------------------
    # ANN support
    # ------------------------------------------------------------------
    def get_vector_measure(self):
        return MEASURE_DOT

    def get_user_vectors(self):
        return np.concatenate(
            [self.user_factors, np.ones((self.user_factors.shape[0], 1))],
            axis=1,
        )

    def get_item_vectors(self):
        item_vec = np.concatenate(
            [self.item_factors, self.item_fm_proj], axis=1
        )
        return np.concatenate(
            [item_vec, self.item_bias.reshape(-1, 1)], axis=1
        )
