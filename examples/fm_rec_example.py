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

"""Example: Feature-Based Foundation Model Recommender (FMRec)
================================================================

This example demonstrates how to use pre-trained text embeddings
(from SentenceTransformers) as item features and train FMRec on
the CiteULike dataset.

Requirements
------------
pip install torch sentence-transformers
"""

import cornac
from cornac.data import TextModality
from cornac.data.transformer_text import TransformerTextModality
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC, Recall

# 1. Load the CiteULike dataset (includes item text)
docs, item_ids = cornac.datasets.citeulike.load_text()
feedback = cornac.datasets.citeulike.load_feedback(reader=cornac.data.Reader())

# 2. Create a TransformerTextModality that encodes text with a
#    SentenceTransformer model and pre-computes embeddings.
item_text_modality = TransformerTextModality(
    corpus=docs,
    ids=item_ids,
    model_name_or_path="paraphrase-MiniLM-L6-v2",
    preencode=True,  # encode entire corpus upfront
)

# 3. Set up the evaluation method with the text modality
eval_method = RatioSplit(
    data=feedback,
    test_size=0.2,
    rating_threshold=0.5,
    item_text=item_text_modality,
    verbose=True,
    seed=42,
)

# 4. Instantiate FMRec and baseline models
fm_rec = cornac.models.FMRec(
    name="FMRec",
    k=50,
    n_epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    verbose=True,
)

bpr = cornac.models.BPR(k=50, max_iter=100, learning_rate=0.01, verbose=True)
most_pop = cornac.models.MostPop()

# 5. Run the experiment
cornac.Experiment(
    eval_method=eval_method,
    models=[fm_rec, bpr, most_pop],
    metrics=[AUC(), Recall(k=50)],
    user_based=True,
).run()
