# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from bert_score import score as bert_score_fn
from utils.prompts import INSTRUCTION_JUDGE, INSTRUCTION_JUDGE_WIN_RATE


def compute_bert_score(generated_responses, golden_answers, lang="en"):
    """Compute BERTScore (Precision, Recall, F1).

    Args:
        generated_responses: List of generated answer strings.
        golden_answers: List of golden answer strings.
        lang: Language code.

    Returns:
        Tuple of (mean_precision, mean_recall, mean_f1).
    """
    P, R, F1 = bert_score_fn(
        generated_responses,
        golden_answers,
        model_type="bert-base-uncased",
        lang=lang,
    )
    return P.mean().item(), R.mean().item(), F1.mean().item()


def compute_llm_judge_accuracy(
    llm_client, questions, generated_responses, golden_answers
):
    """Evaluate correctness using LLM-as-a-Judge.

    Args:
        llm_client: LLMClient instance.
        questions: List of question strings.
        generated_responses: List of generated answers.
        golden_answers: List of golden answers.

    Returns:
        Accuracy (fraction judged as correct).
    """
    correct = 0
    total = len(questions)

    for q, gen, gold in zip(questions, generated_responses, golden_answers):
        prompt = INSTRUCTION_JUDGE.format(question=q, generated=gen, golden=gold)
        judgment = llm_client.generate(prompt, max_new_tokens=10).strip().lower()
        if "correct" in judgment:
            correct += 1

    return correct / total if total > 0 else 0.0


def compute_win_rate(llm_client, questions, answers_a, answers_b):
    """Compute pairwise win rate using LLM-as-a-Judge.

    Args:
        llm_client: LLMClient instance.
        questions: List of question strings.
        answers_a: List of answers from system A.
        answers_b: List of answers from system B.

    Returns:
        Win rate of A over B (fraction where A is judged better).
    """
    wins_a = 0
    total = len(questions)

    for q, a, b in zip(questions, answers_a, answers_b):
        prompt = INSTRUCTION_JUDGE_WIN_RATE.format(question=q, answer_a=a, answer_b=b)
        judgment = llm_client.generate(prompt, max_new_tokens=5).strip().upper()
        if "A" in judgment:
            wins_a += 1

    return wins_a / total if total > 0 else 0.0


def evaluate_generation(llm_client, questions, generated_responses, golden_answers):
    """Run all generation evaluation metrics.

    Args:
        llm_client: LLMClient instance (for LLM-as-Judge).
        questions: List of question strings.
        generated_responses: List of generated answers.
        golden_answers: List of golden answers.

    Returns:
        Dict of metric results.
    """
    p, r, f1 = compute_bert_score(generated_responses, golden_answers)
    accuracy = compute_llm_judge_accuracy(
        llm_client, questions, generated_responses, golden_answers
    )

    return {
        "bert_score": {"precision": p, "recall": r, "f1": f1},
        "llm_judge_accuracy": accuracy,
    }
