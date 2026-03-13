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

"""Example: LLM-based Generative Recommender (LLMRec)
=====================================================

This example demonstrates how to use a causal language model (GPT-2)
to score and generate item recommendations.

LLMRec works by formatting each user's interaction history as a prompt
and scoring candidate items by their log-likelihood under the LLM.

Requirements
------------
pip install torch transformers accelerate
"""

import cornac
from cornac.data import TextModality
from cornac.eval_methods import RatioSplit

# 1. Load a dataset with item text (CiteULike)
docs, item_ids = cornac.datasets.citeulike.load_text()
feedback = cornac.datasets.citeulike.load_feedback(reader=cornac.data.Reader())

# 2. Create a text modality so LLMRec can build an item catalog
item_text_modality = TextModality(corpus=docs, ids=item_ids)

# 3. Set up evaluation
eval_method = RatioSplit(
    data=feedback,
    test_size=0.2,
    rating_threshold=0.5,
    item_text=item_text_modality,
    verbose=True,
    seed=42,
)

# 4. Create the LLMRec model
#    - model_name: any HuggingFace causal LM (GPT-2 is small and fast)
#    - max_history: how many recent interactions to include in the prompt
llm_rec = cornac.models.LLMRec(
    name="LLMRec-GPT2",
    model_name="gpt2",
    max_history=10,
    verbose=True,
)

# 5. Fit (builds item catalog, does NOT fine-tune the LLM)
llm_rec.fit(eval_method.train_set)

# 6. Demonstrate generation mode for a single user
user_idx = 0
print(f"\n--- Generated recommendations for user {user_idx} ---")
recs = llm_rec.generate_recommendations(user_idx, top_k=5)
for i, rec in enumerate(recs, 1):
    print(f"  {i}. {rec}")

# 7. Score a single user-item pair
print(f"\n--- Scoring user {user_idx} on first 5 items ---")
for item_idx in range(min(5, llm_rec.total_items)):
    score = llm_rec.score(user_idx, item_idx)
    title = llm_rec.item_catalog.get(item_idx, f"Item_{item_idx}")[:60]
    print(f"  Item {item_idx} ({title}…): score = {score:.4f}")
