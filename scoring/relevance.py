# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def compute_relevance(query_embedding, utterance_embeddings):
    """Compute relevance scores R(q, u_i) = cos(E_q, E_{u_i}).

    Args:
        query_embedding: 1-D numpy array or tensor of shape (D,).
        utterance_embeddings: dict of {node_id: embedding} or
            list of (node_id, embedding) tuples.

    Returns:
        Dict of {node_id: relevance_score}.
    """
    if isinstance(utterance_embeddings, dict):
        items = list(utterance_embeddings.items())
    else:
        items = list(utterance_embeddings)

    if not items:
        return {}

    node_ids = [item[0] for item in items]
    embs = np.array([item[1] for item in items], dtype=np.float32)

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    emb_norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    scores = emb_norms @ q_norm

    return {nid: float(s) for nid, s in zip(node_ids, scores)}
