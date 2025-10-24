#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --job-name=mem_proj
#SBATCH --gres=gpu:1        # uncomment only if/as needed
##SBATCH --cpus-per-gpu=5    # change as needed
##SBATCH --mem=256000
## %j is the job id, %u is the user id
#SBATCH --output=/home/%u/memory_project/memory-%j.log

nvidia-smi
source ../miniconda3/bin/activate
conda activate mem_llm
# python3 -m run_icl --overlap 10 --chunk_size 50 --max_length 256 --dataset_path datasets/longmemeval_data/longmemeval_s.json --model_path models/Llama-3.3-70B-Instruct
# python3 -m run_icl --chunk_size 512 --dataset_path datasets/longmemeval_data/longmemeval_s.json --model_path models/Llama-3.2-3B-Instruct

# python3 -m run_retrieval --overlap 10 --chunk_size 50 --max_length 256 --dataset_path datasets/longmemeval_data/longmemeval_s.json --model_path models/Llama-3.2-3B-Instruct

# python3 -m run_evaluation --dataset_path datasets/longmemeval_data/longmemeval_m.json
# python3 -m run_evaluation --dataset_path datasets/longmemeval_data/longmemeval_l_100.json

# python3 -m run_memory_generator --dataset_path datasets/longmemeval_data/longmemeval_m.json --model_path models/Llama-3.3-70B-Instruct

# python3 -m run_finetuning --model_path models/Qwen2.5-32B --save_path models/fine-tuned/Qwen2.5-32B-mem-recall-qa --batch_size 4 --num_epochs 10 --learning_rate 2e-5
# python3 -m run_finetuning --model_path models/Qwen2.5-32B --save_path models/fine-tuned/Qwen2.5-32B-mem-recall-qa --batch_size 4 --num_epochs 10 --learning_rate 5e-5
# python3 -m run_finetuning --model_path models/Qwen2.5-32B --save_path models/fine-tuned/Qwen2.5-32B-mem-recall-qa --batch_size 4 --num_epochs 5 --learning_rate 2e-5
# python3 -m run_finetuning --model_path models/Qwen2.5-32B --save_path models/fine-tuned/Qwen2.5-32B-mem-recall-qa --batch_size 4 --num_epochs 5 --learning_rate 5e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 4 --num_epochs 10 --learning_rate 5e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 4 --num_epochs 10 --learning_rate 2e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 4 --num_epochs 5 --learning_rate 5e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 4 --num_epochs 5 --learning_rate 2e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 8 --num_epochs 5 --learning_rate 5e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 8 --num_epochs 10 --learning_rate 2e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 4 --num_epochs 20 --learning_rate 2e-5
python3 -m run_finetuning --model_path models/Llama-3.2-3B-Instruct --save_path models/fine-tuned/Llama-3.2-3B-Instruct-mem-recall-qa --batch_size 8 --num_epochs 20 --learning_rate 2e-5

# python3 -m run_generation --model_name gpt_oss_120B --model_path models/gpt-oss-120b --dataset_path datasets/for_generation_flat_retriever.json