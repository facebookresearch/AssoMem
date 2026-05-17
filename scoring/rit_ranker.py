# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from scoring.importance import compute_importance
from scoring.relevance import compute_relevance
from scoring.temporal import compute_temporal


class RITRanker:
    """Three-signal RIT (Relevance, Importance, Temporal) ranker.

    Combines:
        S(q, u_i) = α_R * R(q, u_i) + α_I * I(q, u_i) + α_T * T(u_i)
    """

    def __init__(self, graph, cmi_fusion=None, decay_constants=None, damping=0.85):
        """
        Args:
            graph: AssociativeMemoryGraph instance.
            cmi_fusion: CMIFusion instance (optional). If None, uses equal weights.
            decay_constants: List of 3 floats for temporal decay.
            damping: PPR damping factor.
        """
        self.graph = graph
        self.cmi_fusion = cmi_fusion
        self.decay_constants = decay_constants
        self.damping = damping

    def rank(
        self,
        query_embedding,
        utterance_nodes,
        utterance_ages=None,
        query_type="unknown",
        top_k=None,
    ):
        """Rank utterance nodes by combined RIT score.

        Args:
            query_embedding: 1-D numpy array query embedding.
            utterance_nodes: List of utterance node IDs to rank.
            utterance_ages: Dict of {node_id: age_in_days}. If None,
                temporal scores default to 1.0 for all.
            query_type: Query type string for CMI weight lookup.
            top_k: If set, return only top-k utterances.

        Returns:
            List of (node_id, combined_score) sorted descending.
        """
        # Gather embeddings for the candidate utterances
        utt_embs = {}
        for node in utterance_nodes:
            emb = self.graph.get_utterance_embedding(node)
            if emb is not None:
                utt_embs[node] = emb

        # Compute relevance
        r_scores = compute_relevance(query_embedding, utt_embs)

        # Compute importance (PPR)
        i_scores = compute_importance(self.graph, query_embedding, damping=self.damping)

        # Compute temporal
        if utterance_ages is None:
            utterance_ages = dict.fromkeys(utterance_nodes)
        t_scores = compute_temporal(utterance_ages, self.decay_constants)

        # Get fusion weights
        if self.cmi_fusion is not None:
            alpha_r, alpha_i, alpha_t = self.cmi_fusion.get_weights(query_type)
        else:
            alpha_r, alpha_i, alpha_t = (1.0 / 3, 1.0 / 3, 1.0 / 3)

        # Combine scores
        combined = {}
        for node in utterance_nodes:
            r = r_scores.get(node, 0.0)
            i = i_scores.get(node, 0.0)
            t = t_scores.get(node, 1.0)
            combined[node] = alpha_r * r + alpha_i * i + alpha_t * t

        # Sort descending
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked
