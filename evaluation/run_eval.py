# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

from evaluation.generation_metrics import evaluate_generation
from evaluation.retrieval_metrics import evaluate_retrieval
from utils.data_loader import load_dataset


def run_retrieval_evaluation(config, retriever):
    """Run retrieval evaluation and print results.

    Args:
        config: Config namespace.
        retriever: CandidateRetriever instance.
    """
    data = load_dataset(config.dataset_name, config.dataset_path)

    k_values = list(range(1, 11))
    results = evaluate_retrieval(
        data, retriever, k_values=k_values, granularity=config.granularity
    )

    print(f"\nRetrieval Evaluation ({config.granularity} granularity):")
    print(f"{'k':<6} {'Recall@k':<12} {'NDCG@k':<12}")
    for k in k_values:
        print(f"{k:<6} {results['recall'][k]:<12.4f} {results['ndcg'][k]:<12.4f}")

    # Save results
    os.makedirs(config.output_path, exist_ok=True)
    output_file = os.path.join(
        config.output_path,
        f"retrieval_eval_{config.granularity}.json",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    return results


def run_generation_evaluation(config, llm_client, questions, generated, golden):
    """Run generation evaluation and print results.

    Args:
        config: Config namespace.
        llm_client: LLMClient instance for LLM-as-Judge.
        questions: List of question strings.
        generated: List of generated answer strings.
        golden: List of golden answer strings.
    """
    results = evaluate_generation(llm_client, questions, generated, golden)

    print("\nGeneration Evaluation:")
    bs = results["bert_score"]
    print(
        f"  BERTScore - P: {bs['precision']:.4f}, R: {bs['recall']:.4f}, F1: {bs['f1']:.4f}"
    )
    print(f"  LLM Judge Accuracy: {results['llm_judge_accuracy']:.4f}")

    os.makedirs(config.output_path, exist_ok=True)
    output_file = os.path.join(config.output_path, "generation_eval.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    return results
