#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=assomem_graph
#SBATCH --output=logs/graph_%j.out
#SBATCH --error=logs/graph_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=04:00:00

python run.py \
    --stage graph \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_name LongMemEval \
    --dataset_path datasets/longmemeval_m.json \
    --embedding_model BAAI/bge-large-en-v1.5 \
    --clue_merge_threshold 0.65 \
    --utterance_sim_threshold 0.75 \
    --graph_save_path results/graph
