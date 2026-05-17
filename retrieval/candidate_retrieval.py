# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class CandidateRetriever:
    """Two-step hybrid retrieval from the associative memory graph.

    Step 1: Retrieve top-K clues by cosine similarity to the query.
    Step 2: Gather all utterances under those clues, score with RITRanker,
            and return top-K utterances.
    """

    def __init__(
        self, graph, embedding_model, rit_ranker, top_k_clues=10, top_k_utterances=6
    ):
        self.graph = graph
        self.embedding_model = embedding_model
        self.rit_ranker = rit_ranker
        self.top_k_clues = top_k_clues
        self.top_k_utterances = top_k_utterances

    def retrieve(self, query, utterance_ages=None, query_type="unknown"):
        """Retrieve top utterances for a query.

        Args:
            query: Query string.
            utterance_ages: Optional dict of {node_id: age_in_days}.
            query_type: Query type string for CMI weight lookup.

        Returns:
            List of (utterance_text, score) tuples, sorted descending.
        """
        # Encode query
        query_emb = self.embedding_model.embed_queries(query)[0]

        # Step 1: Retrieve top-K clues
        clue_nodes = self.graph.get_clue_nodes()
        if not clue_nodes:
            return []

        clue_embs = self.graph.get_all_clue_embeddings()
        clue_ids = list(clue_embs.keys())
        clue_emb_array = np.array([clue_embs[c] for c in clue_ids], dtype=np.float32)

        similarities = self.embedding_model.similarity(query_emb, clue_emb_array)
        top_clue_indices = np.argsort(similarities)[::-1][: self.top_k_clues]
        top_clues = [clue_ids[i] for i in top_clue_indices]

        # Step 2: Gather candidate utterances from top clues
        candidate_utterances = set()
        for clue_node in top_clues:
            utts = self.graph.get_utterances_for_clue(clue_node)
            candidate_utterances.update(utts)

        candidate_utterances = list(candidate_utterances)
        if not candidate_utterances:
            return []

        # Step 3: Score with RITRanker
        ranked = self.rit_ranker.rank(
            query_embedding=query_emb,
            utterance_nodes=candidate_utterances,
            utterance_ages=utterance_ages,
            query_type=query_type,
            top_k=self.top_k_utterances,
        )

        # Convert node IDs to text
        results = []
        for node_id, score in ranked:
            text = self.graph.get_node_text(node_id)
            results.append((text, score))

        return results
