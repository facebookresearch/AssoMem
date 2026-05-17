# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import networkx as nx
import numpy as np
import torch


class AssociativeMemoryGraph:
    """Bipartite associative memory graph G = (V_c ∪ V_u, E_own ∪ E_sim).

    Nodes:
        - Clue nodes (type='clue'): one per merged clue.
        - Utterance nodes (type='utterance'): one per utterance.

    Edges:
        - Ownership edges (type='owns'): clue -> utterance.
        - Similarity edges (type='similar'): utterance <-> utterance
          when cosine similarity > threshold.
    """

    def __init__(self, sim_threshold=0.75):
        self.graph = nx.Graph()
        self.sim_threshold = sim_threshold
        self._clue_embeddings = {}
        self._utterance_embeddings = {}

    def build(self, merged_clues, embedding_model):
        """Build the associative memory graph from merged clues.

        Args:
            merged_clues: List of MergedClue objects.
            embedding_model: EmbeddingModel instance.
        """
        self.graph.clear()
        self._clue_embeddings.clear()
        self._utterance_embeddings.clear()

        # Add clue nodes
        all_clue_texts = [mc.clue_text for mc in merged_clues]
        if all_clue_texts:
            clue_embs = embedding_model.embed_documents(all_clue_texts)

        utterance_id_counter = 0
        all_utterance_texts = []
        utterance_to_clue = {}

        for i, mc in enumerate(merged_clues):
            clue_node = f"clue_{mc.clue_id}"
            clue_emb = clue_embs[i] if all_clue_texts else None
            self.graph.add_node(
                clue_node,
                node_type="clue",
                text=mc.clue_text,
                session_ids=mc.session_ids,
            )
            if clue_emb is not None:
                self._clue_embeddings[clue_node] = clue_emb

            # Add utterance nodes and ownership edges
            for utt_text in mc.utterances:
                utt_node = f"utt_{utterance_id_counter}"
                self.graph.add_node(
                    utt_node,
                    node_type="utterance",
                    text=utt_text,
                )
                self.graph.add_edge(clue_node, utt_node, edge_type="owns")
                all_utterance_texts.append(utt_text)
                utterance_to_clue[utt_node] = clue_node
                utterance_id_counter += 1

        # Embed all utterances
        if all_utterance_texts:
            utt_embs = embedding_model.embed_documents(all_utterance_texts)
            utt_nodes = [f"utt_{i}" for i in range(len(all_utterance_texts))]
            for node, emb in zip(utt_nodes, utt_embs):
                self._utterance_embeddings[node] = emb

            # Add similarity edges between utterances
            emb_tensor = torch.tensor(np.array(utt_embs), dtype=torch.float32)
            normed = emb_tensor / emb_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sim_matrix = torch.mm(normed, normed.t())

            for i in range(len(utt_nodes)):
                for j in range(i + 1, len(utt_nodes)):
                    if sim_matrix[i, j].item() >= self.sim_threshold:
                        self.graph.add_edge(
                            utt_nodes[i],
                            utt_nodes[j],
                            edge_type="similar",
                            weight=sim_matrix[i, j].item(),
                        )

    def get_clue_nodes(self):
        """Return list of clue node IDs."""
        return [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "clue"
        ]

    def get_utterance_nodes(self):
        """Return list of utterance node IDs."""
        return [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "utterance"
        ]

    def get_utterances_for_clue(self, clue_node):
        """Return utterance nodes connected to a clue by ownership edges."""
        neighbors = []
        for neighbor in self.graph.neighbors(clue_node):
            edge_data = self.graph.edges[clue_node, neighbor]
            if edge_data.get("edge_type") == "owns":
                neighbors.append(neighbor)
        return neighbors

    def get_node_text(self, node):
        """Return the text attribute of a node."""
        return self.graph.nodes[node].get("text", "")

    def get_clue_embedding(self, node):
        """Return the embedding for a clue node."""
        return self._clue_embeddings.get(node)

    def get_utterance_embedding(self, node):
        """Return the embedding for an utterance node."""
        return self._utterance_embeddings.get(node)

    def get_all_utterance_embeddings(self):
        """Return dict of {node_id: embedding} for all utterance nodes."""
        return dict(self._utterance_embeddings)

    def get_all_clue_embeddings(self):
        """Return dict of {node_id: embedding} for all clue nodes."""
        return dict(self._clue_embeddings)

    def save(self, path):
        """Save graph and embeddings to disk."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        data = {
            "graph": self.graph,
            "clue_embeddings": self._clue_embeddings,
            "utterance_embeddings": self._utterance_embeddings,
            "sim_threshold": self.sim_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load graph and embeddings from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data["graph"]
        self._clue_embeddings = data["clue_embeddings"]
        self._utterance_embeddings = data["utterance_embeddings"]
        self.sim_threshold = data["sim_threshold"]
