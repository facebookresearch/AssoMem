# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

from retrieval.embedding_model import EmbeddingModel
from tqdm import tqdm
from utils.data_loader import load_dataset
from utils.llm_client import LLMClient
from utils.prompts import INSTRUCTION_GENERATION


def run_rag_baseline(config):
    """Run flat RAG retrieval baseline.

    Retrieves top-k utterances by cosine similarity (no graph, no RIT),
    then generates answers.

    Args:
        config: Config namespace.
    """
    data = load_dataset(config.dataset_name, config.dataset_path)
    embedding_model = EmbeddingModel(config.embedding_model)
    llm = LLMClient(config.model_path, max_length=config.max_input_length)

    top_k = config.top_k_utterances
    results = []

    for sample in tqdm(data, desc="RAG Baseline"):
        question = sample["question"]
        query_emb = embedding_model.embed_queries(question)[0]

        # Collect all utterances
        utterances = []
        for session in sample.get("haystack_sessions", []):
            for turn in session:
                utterances.append(turn["content"])

        if not utterances:
            results.append({"question": question, "output": "IDK"})
            continue

        # Embed and retrieve by cosine similarity
        doc_embs = embedding_model.embed_documents(utterances)
        scores = embedding_model.similarity(query_emb, doc_embs)

        top_indices = scores.argsort()[::-1][:top_k]
        retrieved = [utterances[i] for i in top_indices]

        # Generate
        memory_str = "\n".join(f"Memory evidence: {r}" for r in retrieved)
        prompt = INSTRUCTION_GENERATION.format(memory=memory_str, question=question)
        output = llm.generate(prompt, max_new_tokens=config.max_new_tokens)

        results.append(
            {
                "question": question,
                "retrieved": retrieved,
                "output": output,
            }
        )

    os.makedirs(config.output_path, exist_ok=True)
    output_file = os.path.join(config.output_path, f"{config.model_name}_rag.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"RAG baseline results saved to {output_file}")
    return results
