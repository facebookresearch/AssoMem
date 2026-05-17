# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

from tqdm import tqdm
from utils.data_loader import load_dataset
from utils.llm_client import LLMClient
from utils.prompts import INSTRUCTION_GENERATION


def run_icl_baseline(config):
    """Run in-context learning baseline.

    Feeds all sessions as memory context and generates answers directly.

    Args:
        config: Config namespace with model_path, dataset_path, etc.
    """
    data = load_dataset(config.dataset_name, config.dataset_path)
    llm = LLMClient(config.model_path, max_length=config.max_input_length)

    results = []
    for sample in tqdm(data, desc="ICL Baseline"):
        memory = [str(session) for session in sample["haystack_sessions"]]
        memory_str = "\n".join(memory)
        prompt = INSTRUCTION_GENERATION.format(
            memory=memory_str, question=sample["question"]
        )
        output = llm.generate(prompt, max_new_tokens=config.max_new_tokens)
        results.append(
            {
                "question": sample["question"],
                "output": output,
            }
        )

    os.makedirs(config.output_path, exist_ok=True)
    output_file = os.path.join(config.output_path, f"{config.model_name}_icl.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"ICL baseline results saved to {output_file}")
    return results
