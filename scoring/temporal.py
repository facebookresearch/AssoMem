# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def compute_temporal(utterance_ages, decay_constants=None):
    """Compute temporal scores using multi-scale exponential decay.

    T(u_i) = (exp(-age/τ_1) + exp(-age/τ_2) + exp(-age/τ_3)) / 3

    where τ_1, τ_2, τ_3 are short, medium, and long-term decay constants.

    Args:
        utterance_ages: Dict of {node_id: age_in_days}. Age is the number
            of days between the utterance timestamp and the query time.
            If age is None or negative, score defaults to 1.0.
        decay_constants: List of 3 floats [τ_short, τ_mid, τ_long].
            Defaults to [3.0, 90.0, 365.0].

    Returns:
        Dict of {node_id: temporal_score} in [0, 1].
    """
    if decay_constants is None:
        decay_constants = [3.0, 90.0, 365.0]

    tau_1, tau_2, tau_3 = decay_constants
    scores = {}

    for node_id, age in utterance_ages.items():
        if age is None or age < 0:
            scores[node_id] = 1.0
            continue

        x = float(age)
        score = (
            math.exp(-x / tau_1) + math.exp(-x / tau_2) + math.exp(-x / tau_3)
        ) / 3.0
        scores[node_id] = score

    return scores
