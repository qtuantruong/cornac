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

"""LLM-based Generative Recommender (LLMRec).

Frames recommendation as a **language generation** task.  User interaction
history is formatted as a natural-language prompt and fed to a causal
language model (GPT-2, LLaMA, Mistral, …).  Item scores are derived from
the log-likelihood the LLM assigns to each candidate item title.

This model implements the **Generative paradigm** described in the
FM4RecSys survey (arXiv 2504.16420) and is conceptually aligned with the
Netflix Generative Recommender architecture (next-event prediction via
transformers).

Two scoring modes are supported:

* **Likelihood scoring** (default) — compute the conditional probability
  of each candidate item title given the user prompt; rank by
  log-likelihood.
* **Generation mode** — let the LLM freely generate recommendations as
  text; parse item IDs from the generated output.
"""

from typing import Dict, List, Optional

import numpy as np

from ..recommender import Recommender
from ...exception import CornacException


# =====================================================================
# Default prompt template
# =====================================================================
DEFAULT_PROMPT_TEMPLATE = (
    "A user has interacted with the following items (from earliest to most recent):\n"
    "{history}\n\n"
    "Based on this interaction history, the user would most likely interact with: "
)


class LLMRec(Recommender):
    """LLM-based Generative Recommender.

    Parameters
    ----------
    name : str, default ``'LLMRec'``
    model_name : str, default ``'gpt2'``
        Any HuggingFace ``AutoModelForCausalLM`` identifier.
    prompt_template : str or None
        A Python format-string with a ``{history}`` placeholder.
        If *None*, :data:`DEFAULT_PROMPT_TEMPLATE` is used.
    max_history : int, default 20
        Maximum number of recent interactions to include in prompt.
    max_new_tokens : int, default 64
        Max tokens for generation mode.
    device : str or None
        PyTorch device string.
    item_catalog : dict or None
        Mapping from **item index** → **item title string**.
        If not provided, the model will attempt to derive titles from
        ``train_set.item_text`` modality.
    verbose : bool, default True
    """

    def __init__(
        self,
        name="LLMRec",
        model_name="gpt2",
        prompt_template=None,
        max_history=20,
        max_new_tokens=64,
        device=None,
        item_catalog=None,
        verbose=True,
    ):
        super().__init__(name=name, trainable=False, verbose=verbose)
        self.model_name = model_name
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.max_history = max_history
        self.max_new_tokens = max_new_tokens
        self._device = device
        self.item_catalog: Optional[Dict[int, str]] = item_catalog

        # Loaded lazily
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._device = device

        if self.verbose:
            print(f"[LLMRec] Loading '{self.model_name}' on {device}…")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        ).to(device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Fit (catalog construction)
    # ------------------------------------------------------------------
    def fit(self, train_set, val_set=None):
        """Build the item catalog from ``train_set``.

        The LLM itself is **not** fine-tuned here (set ``trainable=False``).
        If you need fine-tuning, subclass and override ``_fit_finetune``.
        """
        Recommender.fit(self, train_set, val_set)

        # Build item catalog if not provided
        if self.item_catalog is None:
            self.item_catalog = self._build_catalog(train_set)
            if self.item_catalog is None:
                raise CornacException(
                    "LLMRec requires an item_catalog (item_idx → title) "
                    "or an item_text modality with a corpus."
                )

        # Build reverse map: item index → raw item_id
        self._idx_to_id = {v: k for k, v in self.iid_map.items()}

        return self

    def _build_catalog(self, train_set) -> Optional[Dict[int, str]]:
        """Attempt to build catalog from train_set modalities."""
        item_text = getattr(train_set, "item_text", None)
        if item_text is None:
            return None

        corpus = getattr(item_text, "corpus", None)
        if corpus is None:
            return None

        catalog = {}
        ids = getattr(item_text, "ids", None)
        if ids is not None:
            for raw_id, text in zip(ids, corpus):
                idx = self.iid_map.get(raw_id)
                if idx is not None:
                    catalog[idx] = text
        else:
            for idx, text in enumerate(corpus):
                if idx < self.total_items:
                    catalog[idx] = text
        return catalog

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(self, user_idx) -> str:
        """Create the LLM prompt for a user from interaction history."""
        user_items = self.train_set.user_data.get(user_idx, ([], []))
        item_indices = user_items[0][-self.max_history :]

        history_lines = []
        for idx in item_indices:
            title = self.item_catalog.get(idx, f"Item_{idx}")
            history_lines.append(f"- {title}")

        history_str = "\n".join(history_lines) if history_lines else "- (no history)"
        return self.prompt_template.format(history=history_str)

    # ------------------------------------------------------------------
    # Scoring via log-likelihood
    # ------------------------------------------------------------------
    def score(self, user_idx, item_idx=None):
        """Score items for a user via LLM log-likelihood.

        For each candidate item, we compute the conditional log-probability
        of the item's title given the user's prompt context.
        """
        import torch

        self._ensure_model()

        prompt = self._build_prompt(user_idx)

        if item_idx is not None:
            return self._score_single(prompt, item_idx)

        # Score all known items
        scores = np.full(self.total_items, -1e9)
        for idx in range(self.total_items):
            if idx in self.item_catalog:
                scores[idx] = self._score_single(prompt, idx)
        return scores

    def _score_single(self, prompt: str, item_idx: int) -> float:
        """Compute log-likelihood score for a single item."""
        import torch

        title = self.item_catalog.get(item_idx, f"Item_{item_idx}")
        full_text = prompt + title

        inputs = self._tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)

        prompt_inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])

        # Extract log-probs only for the title portion
        logits = outputs.logits[0, prompt_len - 1 : -1, :]
        target_ids = inputs["input_ids"][0, prompt_len:]

        if len(target_ids) == 0:
            return -1e9

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(len(target_ids)), target_ids
        ]
        # Average log-prob (length-normalised)
        return float(token_log_probs.mean().cpu())

    # ------------------------------------------------------------------
    # Generation mode
    # ------------------------------------------------------------------
    def generate_recommendations(
        self,
        user_idx: int,
        top_k: int = 10,
    ) -> List[str]:
        """Generate free-text recommendations for a user.

        Returns
        -------
        recommendations : list of str
            Raw generated text (may need parsing to extract item names).
        """
        import torch

        self._ensure_model()
        prompt = self._build_prompt(user_idx)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return [line.strip("- ").strip() for line in generated.split("\n") if line.strip()][:top_k]
