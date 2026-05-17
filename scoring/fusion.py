# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter

import numpy as np


def _bin_scores(scores, n_bins=3):
    """Bin scores into discrete levels (low/medium/high).

    Args:
        scores: List of float scores.
        n_bins: Number of bins (default 3).

    Returns:
        List of bin indices (0=low, 1=medium, 2=high).
    """
    if not scores:
        return []
    sorted_scores = sorted(scores)
    thresholds = [
        sorted_scores[int(len(sorted_scores) * (i + 1) / n_bins) - 1]
        for i in range(n_bins - 1)
    ]

    binned = []
    for s in scores:
        b = 0
        for t in thresholds:
            if s > t:
                b += 1
        binned.append(min(b, n_bins - 1))
    return binned


def _compute_cmi(signal_bins, answer_labels, query_types, n_bins=3):
    """Compute Conditional Mutual Information I(Signal; Answer | QueryType).

    CMI = Σ_{q,s,a} P(q,s,a) * log( P(s,a|q) / (P(s|q)*P(a|q)) )

    Args:
        signal_bins: List of binned signal values.
        answer_labels: List of binary labels (1=correct, 0=incorrect).
        query_types: List of query type strings.
        n_bins: Number of signal bins.

    Returns:
        Float CMI value.
    """
    n = len(signal_bins)
    if n == 0:
        return 0.0

    # Count joint and marginal occurrences
    count_qsa: Counter = Counter()
    count_qs: Counter = Counter()
    count_qa: Counter = Counter()
    count_q: Counter = Counter()

    for s, a, q in zip(signal_bins, answer_labels, query_types):
        count_qsa[(q, s, a)] += 1
        count_qs[(q, s)] += 1
        count_qa[(q, a)] += 1
        count_q[q] += 1

    cmi = 0.0
    for (q, s, a), count in count_qsa.items():
        p_qsa = count / n
        p_sa_given_q = count / count_q[q]
        p_s_given_q = count_qs[(q, s)] / count_q[q]
        p_a_given_q = count_qa[(q, a)] / count_q[q]

        denom = p_s_given_q * p_a_given_q
        if denom > 0 and p_sa_given_q > 0:
            cmi += p_qsa * math.log(p_sa_given_q / denom)

    return cmi


class CMIFusion:
    """CMI-based adaptive fusion of R, I, T signals.

    Computes per-query-type weights using Conditional Mutual Information,
    then applies softmax with temperature to get α_R, α_I, α_T.
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.weights = {}  # {query_type: (α_R, α_I, α_T)}

    def fit(
        self,
        relevance_scores,
        importance_scores,
        temporal_scores,
        answer_labels,
        query_types,
    ):
        """Learn CMI-based weights from labeled data.

        Args:
            relevance_scores: List of relevance scores per sample.
            importance_scores: List of importance scores per sample.
            temporal_scores: List of temporal scores per sample.
            answer_labels: List of binary labels (1 if utterance is relevant).
            query_types: List of query type strings per sample.
        """
        r_bins = _bin_scores(relevance_scores)
        i_bins = _bin_scores(importance_scores)
        t_bins = _bin_scores(temporal_scores)

        unique_qtypes = set(query_types)

        for qt in unique_qtypes:
            # Filter samples of this query type
            indices = [idx for idx, q in enumerate(query_types) if q == qt]
            if not indices:
                continue

            qt_answers = [answer_labels[idx] for idx in indices]

            cmi_r = _compute_cmi(
                [r_bins[idx] for idx in indices], qt_answers, [qt] * len(indices)
            )
            cmi_i = _compute_cmi(
                [i_bins[idx] for idx in indices], qt_answers, [qt] * len(indices)
            )
            cmi_t = _compute_cmi(
                [t_bins[idx] for idx in indices], qt_answers, [qt] * len(indices)
            )

            # Softmax with temperature
            raw = np.array([cmi_r, cmi_i, cmi_t]) / self.temperature
            raw = raw - raw.max()  # numerical stability
            exp_vals = np.exp(raw)
            weights = exp_vals / exp_vals.sum()

            self.weights[qt] = (float(weights[0]), float(weights[1]), float(weights[2]))

        # Default weights for unseen query types
        if self.weights:
            avg = np.mean([list(w) for w in self.weights.values()], axis=0)
            self.weights["_default"] = tuple(avg.tolist())
        else:
            self.weights["_default"] = (1.0 / 3, 1.0 / 3, 1.0 / 3)

    def get_weights(self, query_type):
        """Get fusion weights for a query type.

        Returns:
            Tuple of (α_R, α_I, α_T).
        """
        return self.weights.get(
            query_type, self.weights.get("_default", (1 / 3, 1 / 3, 1 / 3))
        )
