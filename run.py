# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AssoMem: Main pipeline entry point.

Runs the full AssoMem pipeline or individual stages:
    1. Graph construction (clue tagging → merging → associative graph)
    2. Retrieval (two-step hybrid with RIT ranking)
    3. Generation (answer generation from retrieved context)
    4. Finetuning (QLoRA multi-task finetuning)
    5. Evaluation (retrieval + generation metrics)
"""

import json
import os

from config import get_config
from generation.answer_generator import AnswerGenerator
from graph.associative_graph import AssociativeMemoryGraph
from graph.clue_merger import ClueMerger
from graph.clue_tagger import ClueTagger
from retrieval.candidate_retrieval import CandidateRetriever
from retrieval.embedding_model import EmbeddingModel
from scoring.fusion import CMIFusion
from scoring.rit_ranker import RITRanker
from tqdm import tqdm
from utils.data_loader import load_dataset
from utils.llm_client import LLMClient


def build_graph(config):
    """Stage 1: Build the associative memory graph."""
    print("=== Stage 1: Graph Construction ===")
    data = load_dataset(config.dataset_name, config.dataset_path)
    llm = LLMClient(config.model_path)
    embedding_model = EmbeddingModel(config.embedding_model)

    tagger = ClueTagger(llm)
    merger = ClueMerger(embedding_model, threshold=config.clue_merge_threshold)

    all_clue_tags = []
    for sample in tqdm(data, desc="Clue tagging"):
        sessions = sample.get("haystack_sessions", [])
        session_ids = sample.get(
            "haystack_session_ids", [str(i) for i in range(len(sessions))]
        )
        clue_tags = tagger.tag_all_sessions(sessions, session_ids)
        all_clue_tags.extend(clue_tags)

    print(f"Tagged {len(all_clue_tags)} sessions")

    merged_clues = merger.merge(all_clue_tags)
    print(f"Merged into {len(merged_clues)} clue groups")

    graph = AssociativeMemoryGraph(sim_threshold=config.utterance_sim_threshold)
    graph.build(merged_clues, embedding_model)
    print(
        f"Graph: {len(graph.get_clue_nodes())} clue nodes, "
        f"{len(graph.get_utterance_nodes())} utterance nodes, "
        f"{graph.graph.number_of_edges()} edges"
    )

    os.makedirs(config.graph_save_path, exist_ok=True)
    graph_path = os.path.join(config.graph_save_path, "assomem_graph.pkl")
    graph.save(graph_path)
    print(f"Graph saved to {graph_path}")

    return graph


def run_retrieval(config, graph=None):
    """Stage 2: Retrieve memories for all queries."""
    print("=== Stage 2: Retrieval ===")
    if graph is None:
        graph = AssociativeMemoryGraph()
        graph.load(os.path.join(config.graph_save_path, "assomem_graph.pkl"))

    embedding_model = EmbeddingModel(config.embedding_model)

    cmi_fusion = CMIFusion(temperature=config.cmi_temperature)
    # If no pre-trained weights, use equal weights
    rit_ranker = RITRanker(
        graph=graph,
        cmi_fusion=cmi_fusion,
        decay_constants=config.temporal_decay_weights,
        damping=config.ppr_damping,
    )
    retriever = CandidateRetriever(
        graph=graph,
        embedding_model=embedding_model,
        rit_ranker=rit_ranker,
        top_k_clues=config.top_k_clues,
        top_k_utterances=config.top_k_utterances,
    )

    data = load_dataset(config.dataset_name, config.dataset_path)
    results = []

    for sample in tqdm(data, desc="Retrieving"):
        question = sample["question"]
        retrieved = retriever.retrieve(question)
        results.append(
            {
                "question": question,
                "retrieved": [
                    {"text": text, "score": score} for text, score in retrieved
                ],
            }
        )

    os.makedirs(config.output_path, exist_ok=True)
    output_file = os.path.join(config.output_path, "retrieval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Retrieval results saved to {output_file}")
    return results, retriever


def run_generation(config, retrieval_results=None):
    """Stage 3: Generate answers from retrieved context."""
    print("=== Stage 3: Generation ===")
    llm = LLMClient(config.model_path, max_length=config.max_input_length)
    generator = AnswerGenerator(llm, multitask=True)

    if retrieval_results is None:
        retrieval_path = os.path.join(config.output_path, "retrieval_results.json")
        with open(retrieval_path, "r", encoding="utf-8") as f:
            retrieval_results = json.load(f)

    results = []
    for item in tqdm(retrieval_results, desc="Generating"):
        retrieved = [(r["text"], r["score"]) for r in item["retrieved"]]
        answer = generator.generate(
            item["question"], retrieved, max_new_tokens=config.max_new_tokens
        )
        results.append(
            {
                "question": item["question"],
                "output": answer,
            }
        )

    output_file = os.path.join(config.output_path, "generation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Generation results saved to {output_file}")
    return results


def run_finetune(config):
    """Stage 4: QLoRA finetuning."""
    print("=== Stage 4: Finetuning ===")
    from generation.finetuning import run_finetuning

    run_finetuning(config)


def run_evaluate(config):
    """Stage 5: Evaluation."""
    print("=== Stage 5: Evaluation ===")
    from evaluation.run_eval import run_generation_evaluation, run_retrieval_evaluation

    # Load graph and build retriever for retrieval eval
    graph = AssociativeMemoryGraph()
    graph.load(os.path.join(config.graph_save_path, "assomem_graph.pkl"))
    embedding_model = EmbeddingModel(config.embedding_model)
    rit_ranker = RITRanker(
        graph=graph,
        decay_constants=config.temporal_decay_weights,
        damping=config.ppr_damping,
    )
    retriever = CandidateRetriever(
        graph=graph,
        embedding_model=embedding_model,
        rit_ranker=rit_ranker,
        top_k_clues=config.top_k_clues,
        top_k_utterances=config.top_k_utterances,
    )
    run_retrieval_evaluation(config, retriever)

    # Generation eval if results exist
    gen_path = os.path.join(config.output_path, "generation_results.json")
    if os.path.exists(gen_path):
        with open(gen_path, "r") as f:
            gen_results = json.load(f)

        data = load_dataset(config.dataset_name, config.dataset_path)
        questions = [r["question"] for r in gen_results]
        generated = [r["output"] for r in gen_results]
        golden = [
            d.get("answer", d.get("golden_answer", "")) for d in data[: len(generated)]
        ]

        llm = LLMClient(config.model_path)
        run_generation_evaluation(config, llm, questions, generated, golden)


def main():
    config = get_config()

    if config.stage == "graph":
        build_graph(config)
    elif config.stage == "retrieve":
        run_retrieval(config)
    elif config.stage == "generate":
        run_generation(config)
    elif config.stage == "finetune":
        run_finetune(config)
    elif config.stage == "evaluate":
        run_evaluate(config)
    elif config.stage == "pipeline":
        graph = build_graph(config)
        retrieval_results, _ = run_retrieval(config, graph)
        run_generation(config, retrieval_results)
        run_evaluate(config)


if __name__ == "__main__":
    main()
