# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_config():
    parser = argparse.ArgumentParser(description="AssoMem Configuration")

    # --- Mode ---
    parser.add_argument(
        "--stage",
        type=str,
        default="pipeline",
        choices=["graph", "retrieve", "generate", "finetune", "evaluate", "pipeline"],
        help="Which stage to run",
    )

    # --- Model ---
    parser.add_argument("--model_name", type=str, default="Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--model_path", type=str, default="meta-llama/Llama-3.3-70B-Instruct"
    )

    # --- Dataset ---
    parser.add_argument("--dataset_name", type=str, default="LongMemEval")
    parser.add_argument("--dataset_path", type=str, required=False)

    # --- Embedding ---
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")

    # --- Graph construction ---
    parser.add_argument(
        "--clue_merge_threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for merging clues",
    )
    parser.add_argument(
        "--utterance_sim_threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for utterance-utterance edges",
    )

    # --- Scoring ---
    parser.add_argument(
        "--ppr_damping",
        type=float,
        default=0.85,
        help="Damping factor for Personalized PageRank",
    )
    parser.add_argument(
        "--temporal_decay_weights",
        type=float,
        nargs=3,
        default=[3.0, 90.0, 365.0],
        help="Exponential decay time constants (short, mid, long)",
    )
    parser.add_argument(
        "--cmi_temperature",
        type=float,
        default=1.0,
        help="Temperature for CMI softmax weight assignment",
    )

    # --- Retrieval ---
    parser.add_argument(
        "--top_k_clues",
        type=int,
        default=10,
        help="Number of top clues to retrieve in first stage",
    )
    parser.add_argument(
        "--top_k_utterances",
        type=int,
        default=6,
        help="Number of top utterances to return after RIT ranking",
    )
    parser.add_argument(
        "--granularity", type=str, default="utterance", choices=["utterance", "session"]
    )

    # --- Generation ---
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=4096)

    # --- Finetuning (QLoRA) ---
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="models/fine-tuned/assomem")

    # --- Output ---
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--graph_save_path", type=str, default="results/graph")

    return parser.parse_args()
