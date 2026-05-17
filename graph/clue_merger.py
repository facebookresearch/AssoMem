# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch


@dataclass
class MergedClue:
    """A merged clue grouping multiple sessions under one topic."""

    clue_id: int
    clue_text: str
    session_ids: list = field(default_factory=list)
    utterances: list = field(default_factory=list)


class ClueMerger:
    """Merges similar clues by pairwise cosine similarity thresholding.

    Implements the clue merging step from Section 3.1:
    clues with cosine similarity > delta are merged, and their
    sessions/utterances are grouped together.
    """

    def __init__(self, embedding_model, threshold=0.65):
        self.embedding_model = embedding_model
        self.threshold = threshold

    def merge(self, clue_tags):
        """Merge similar clues.

        Args:
            clue_tags: List of ClueTag objects from ClueTagger.

        Returns:
            List of MergedClue objects.
        """
        if not clue_tags:
            return []

        clue_texts = [ct.clue_text for ct in clue_tags]
        embeddings = torch.tensor(
            self.embedding_model.embed_documents(clue_texts), dtype=torch.float32
        )

        # Compute pairwise cosine similarity matrix
        normed = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim_matrix = torch.mm(normed, normed.t())

        # Greedy grouping: assign each clue to the first group it matches
        n = len(clue_tags)
        assigned = [False] * n
        groups = []

        for i in range(n):
            if assigned[i]:
                continue
            group_indices = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                if sim_matrix[i, j].item() >= self.threshold:
                    group_indices.append(j)
                    assigned[j] = True
            groups.append(group_indices)

        # Build MergedClue objects
        merged_clues = []
        for gid, indices in enumerate(groups):
            # Use the first clue's text as the representative
            representative_text = clue_tags[indices[0]].clue_text
            session_ids = []
            utterances = []
            for idx in indices:
                session_ids.append(clue_tags[idx].session_id)
                utterances.extend(clue_tags[idx].utterances)
            merged_clues.append(
                MergedClue(
                    clue_id=gid,
                    clue_text=representative_text,
                    session_ids=session_ids,
                    utterances=utterances,
                )
            )

        return merged_clues
