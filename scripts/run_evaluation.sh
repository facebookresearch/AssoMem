#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=assomem_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=04:00:00

python run.py \
    --stage evaluate \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_name LongMemEval \
    --dataset_path datasets/longmemeval_m.json \
    --embedding_model BAAI/bge-large-en-v1.5 \
    --graph_save_path results/graph \
    --top_k_clues 10 \
    --top_k_utterances 6 \
    --granularity utterance \
    --output_path results
