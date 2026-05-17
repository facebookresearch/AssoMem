# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import networkx as nx
import numpy as np


def compute_importance(graph, query_embedding, damping=0.85):
    """Compute importance scores via Personalized PageRank on the associative memory graph.

    The personalization (teleportation) vector is:
        - For utterance nodes: cos(E_q, E_{u_i})  (clamped to non-negative)
        - For clue nodes: 0

    Args:
        graph: AssociativeMemoryGraph instance.
        query_embedding: 1-D numpy array of shape (D,).
        damping: Damping factor for PageRank (default 0.85).

    Returns:
        Dict of {utterance_node_id: importance_score}.
    """
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-8)

    # Build personalization vector
    personalization = {}
    for node in graph.graph.nodes():
        node_data = graph.graph.nodes[node]
        if node_data.get("node_type") == "utterance":
            emb = graph.get_utterance_embedding(node)
            if emb is not None:
                emb = np.array(emb, dtype=np.float32)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(q_norm, emb_norm))
                personalization[node] = max(sim, 0.0)
            else:
                personalization[node] = 0.0
        else:
            # Clue nodes get 0 personalization
            personalization[node] = 0.0

    # Ensure at least one non-zero value to avoid division by zero in PageRank
    total = sum(personalization.values())
    if total == 0:
        # Fallback: uniform over utterance nodes
        utt_nodes = graph.get_utterance_nodes()
        if utt_nodes:
            for n in utt_nodes:
                personalization[n] = 1.0 / len(utt_nodes)

    ppr_scores = nx.pagerank(
        graph.graph,
        alpha=damping,
        personalization=personalization,
    )

    # Return only utterance node scores
    return {
        node: score
        for node, score in ppr_scores.items()
        if graph.graph.nodes[node].get("node_type") == "utterance"
    }
