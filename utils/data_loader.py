# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json


def load_dataset(dataset_name, dataset_path):
    """Load a dataset from a JSON file.

    Args:
        dataset_name: Name of the dataset (for format-specific handling).
        dataset_path: Path to the JSON file.

    Returns:
        List of data samples.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_sessions(data_point):
    """Extract sessions from a single data point.

    Returns:
        List of sessions, where each session is a list of turn dicts.
    """
    return data_point.get("haystack_sessions", [])


def extract_utterances(data_point, granularity="utterance"):
    """Extract utterances from a data point at the given granularity.

    Args:
        data_point: Single data sample dict.
        granularity: 'utterance' for individual turns, 'session' for full sessions.

    Returns:
        List of (text, has_answer, session_id) tuples.
    """
    sessions = data_point.get("haystack_sessions", [])
    session_ids = data_point.get("haystack_session_ids", [])

    results = []
    if granularity == "utterance":
        for sid, session in zip(session_ids, sessions):
            for turn in session:
                has_answer = turn.get("has_answer", False)
                results.append((turn["content"], has_answer, sid))
    elif granularity == "session":
        for sid, session in zip(session_ids, sessions):
            text = " ".join(turn["content"] for turn in session)
            has_answer = "answer" in sid
            results.append((text, has_answer, sid))
    return results


def extract_question(data_point):
    """Extract the question from a data point."""
    return data_point["question"]


def extract_golden_answer(data_point):
    """Extract golden answer(s) from a data point."""
    return data_point.get("answer", data_point.get("golden_answer", ""))


def extract_question_type(data_point):
    """Extract question type if available."""
    return data_point.get("question_type", "unknown")
