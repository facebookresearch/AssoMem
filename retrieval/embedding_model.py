# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from FlagEmbedding import FlagAutoModel


class EmbeddingModel:
    """Unified embedding interface using BGE (FlagEmbedding).

    Wraps FlagAutoModel for query and document encoding with a
    consistent API used by all AssoMem components.
    """

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", use_fp16=True):
        self.model = FlagAutoModel.from_finetuned(
            model_name,
            query_instruction_for_retrieval=(
                "Given a question, you need to retrieve relevant "
                "information for answering the question."
            ),
            use_fp16=use_fp16,
        )

    def embed_queries(self, texts):
        """Encode query texts.

        Args:
            texts: str or list of str.

        Returns:
            numpy array of shape (N, D).
        """
        if isinstance(texts, str):
            texts = [texts]
        return np.array(self.model.encode_queries(texts), dtype=np.float32)

    def embed_documents(self, texts):
        """Encode document/passage texts.

        Args:
            texts: str or list of str.

        Returns:
            numpy array of shape (N, D).
        """
        if isinstance(texts, str):
            texts = [texts]
        return np.array(self.model.encode(texts), dtype=np.float32)

    def similarity(self, query_emb, doc_embs):
        """Compute cosine similarity between a query and document embeddings.

        Args:
            query_emb: 1-D array of shape (D,).
            doc_embs: 2-D array of shape (N, D).

        Returns:
            1-D array of N cosine similarity scores.
        """
        q = np.array(query_emb, dtype=np.float32)
        d = np.array(doc_embs, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        d_norms = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
        return d_norms @ q_norm
