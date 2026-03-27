# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
from config import get_config
from utils.prompts import INSTRUCTION_GENERATION
from utils.dataset_process import DatasetProcessor

def generate(config, data):
    # Initialize the Accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token=tokenizer.eos_token

    # Prepare model for distributed inference
    model = accelerator.prepare(model)

    results = []

    with tqdm(total=len(data), desc="Processing") as pbar:
        for pair in data:
            # Use all sessions as memory
            memory = [str(session) for session in pair["haystack_sessions"]]
            input_text = INSTRUCTION_GENERATION.format(memory=memory, question=pair["question"])

            # Tokenize the full input, then chunk by token count
            full_input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=False)
            token_chunks = []
            for i in range(0, full_input_ids.size(1), config.chunk_size):
                end = min(i + config.chunk_size, full_input_ids.size(1))
                token_chunks.append(full_input_ids[:, i:end])

            generated_text = ""
            for chunk_ids in token_chunks:
                chunk_ids = chunk_ids.to(accelerator.device)
                outputs = model.generate(chunk_ids, max_length=config.max_length, num_return_sequences=1)
                generated_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
                chunk_text = tokenizer.decode(chunk_ids[0], skip_special_tokens=True)
                generated_text += generated_chunk.replace(chunk_text, "").strip()
                gpu_info = torch.cuda.memory_allocated("cuda")
                print(f"GPU Memory Allocated: {gpu_info / (1024 ** 2):.2f} MB")

            results.append({"input": pair["question"], "output": generated_text})
            pbar.update(1)

    return results

def main():
    config = get_config()
    data_processor = DatasetProcessor(dataset_name=config.dataset_name, data_path=config.dataset_path)
    data = data_processor.load_dataset()

    # Run the generate function
    results = generate(config, data)

    # Save results or further processing
    path=os.path.join(config.output_path, "{}_icl.json".format(config.model_name))
    with open(path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Generated results saved to {path}")

if __name__ == "__main__":
    main()