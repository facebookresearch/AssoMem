#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=assomem_retrieve
#SBATCH --output=logs/retrieve_%j.out
#SBATCH --error=logs/retrieve_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

python run.py \
    --stage retrieve \
    --dataset_name LongMemEval \
    --dataset_path datasets/longmemeval_m.json \
    --embedding_model BAAI/bge-large-en-v1.5 \
    --graph_save_path results/graph \
    --top_k_clues 10 \
    --top_k_utterances 6 \
    --ppr_damping 0.85 \
    --output_path results
