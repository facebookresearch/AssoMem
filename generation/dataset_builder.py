# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random


class DenoisingDatasetBuilder:
    """Build denoising QA training dataset.

    Creates training examples with a mix of golden (relevant) and
    retrieved (potentially noisy) memories as context, following
    Section 3.3 of the paper.
    """

    def __init__(self, retriever, num_augmentations=6, max_retrieved=6, seed=42):
        """
        Args:
            retriever: CandidateRetriever instance for retrieving noisy memories.
            num_augmentations: Number of augmented examples per query.
            max_retrieved: Number of retrieved memories to mix in.
            seed: Random seed.
        """
        self.retriever = retriever
        self.num_augmentations = num_augmentations
        self.max_retrieved = max_retrieved
        self.rng = random.Random(seed)

    def build_dataset(self, data, question_type_key="question_type"):
        """Build denoising dataset from raw data.

        Args:
            data: List of data samples, each with 'question', 'answer',
                and session fields.
            question_type_key: Key for question type in data samples.

        Returns:
            List of training dicts with keys:
                'query', 'answer', 'context', 'question_type'.
        """
        training_samples = []

        for sample in data:
            question = sample["question"]
            answer = sample.get("answer", sample.get("golden_answer", ""))
            q_type = sample.get(question_type_key, "unknown")

            # Collect golden memories (utterances with has_answer=True)
            golden_memories = []
            for session in sample.get("haystack_sessions", []):
                for turn in session:
                    if turn.get("has_answer", False):
                        golden_memories.append(turn["content"])

            # Retrieve potentially noisy memories
            retrieved = self.retriever.retrieve(question)
            retrieved_texts = [text for text, _ in retrieved[: self.max_retrieved]]

            # Create augmented training examples
            for _ in range(self.num_augmentations):
                # Mix golden and retrieved memories in random order
                context_pool = list(golden_memories) + list(retrieved_texts)
                self.rng.shuffle(context_pool)

                # Subsample to limit context size
                context_size = self.rng.randint(
                    max(1, len(golden_memories)),
                    len(context_pool),
                )
                context = context_pool[:context_size]

                training_samples.append(
                    {
                        "query": question,
                        "answer": f"[TYPE: {q_type}] {answer}",
                        "context": "\n".join(
                            f"Memory evidence: {mem}" for mem in context
                        ),
                        "question_type": q_type,
                    }
                )

        return training_samples

    def save_dataset(self, samples, path):
        """Save training dataset to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    def load_dataset(self, path):
        """Load training dataset from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
