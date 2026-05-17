# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def calculate_recall(predictions, relevant_items, top_k):
    """Calculate Recall@k.

    Args:
        predictions: Ordered list of predicted items.
        relevant_items: Set or list of ground-truth relevant items.
        top_k: Number of top predictions to consider.

    Returns:
        Recall@k score.
    """
    relevant_set = set(relevant_items)
    retrieved_relevant = sum(1 for p in predictions[:top_k] if p in relevant_set)
    total_relevant = len(relevant_set)
    return retrieved_relevant / total_relevant if total_relevant > 0 else 0.0


def calculate_ndcg(predictions, relevant_items, top_k):
    """Calculate NDCG@k.

    Args:
        predictions: Ordered list of predicted items.
        relevant_items: Set or list of ground-truth relevant items.
        top_k: Number of top predictions to consider.

    Returns:
        NDCG@k score.
    """
    relevant_set = set(relevant_items)

    dcg = sum(
        1.0 / math.log2(idx + 2)
        for idx, p in enumerate(predictions[:top_k])
        if p in relevant_set
    )

    ideal_hits = min(len(relevant_set), top_k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(dataset, retriever, k_values=None, granularity="utterance"):
    """Evaluate retrieval performance across a dataset.

    Args:
        dataset: List of data samples.
        retriever: CandidateRetriever instance.
        k_values: List of k values for Recall@k and NDCG@k.
        granularity: 'utterance' or 'session'.

    Returns:
        Dict of {metric_name: {k: average_score}}.
    """
    if k_values is None:
        k_values = list(range(1, 11))

    recall_results = {k: [] for k in k_values}
    ndcg_results = {k: [] for k in k_values}

    for sample in dataset:
        question = sample["question"]

        # Retrieve
        retrieved = retriever.retrieve(question)
        retrieved_texts = [text for text, _ in retrieved]

        # Collect ground-truth relevant items
        relevant = []
        if granularity == "utterance":
            for session in sample.get("haystack_sessions", []):
                for turn in session:
                    if turn.get("has_answer", False):
                        relevant.append(turn["content"])
        elif granularity == "session":
            for sid, session in zip(
                sample.get("haystack_session_ids", []),
                sample.get("haystack_sessions", []),
            ):
                if "answer" in sid:
                    relevant.append(" ".join(t["content"] for t in session))

        for k in k_values:
            recall_results[k].append(calculate_recall(retrieved_texts, relevant, k))
            ndcg_results[k].append(calculate_ndcg(retrieved_texts, relevant, k))

    avg_recall = {
        k: sum(scores) / len(scores) if scores else 0.0
        for k, scores in recall_results.items()
    }
    avg_ndcg = {
        k: sum(scores) / len(scores) if scores else 0.0
        for k, scores in ndcg_results.items()
    }

    return {"recall": avg_recall, "ndcg": avg_ndcg}
