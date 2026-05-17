#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=assomem_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00

python run.py \
    --stage finetune \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path datasets/curated_data.json \
    --save_path models/fine-tuned/assomem \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 3
