# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMClient:
    """Unified LLM inference client using HuggingFace transformers."""

    def __init__(
        self, model_path, device_map="auto", torch_dtype=None, max_length=4096
    ):
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length

    def generate(self, prompt, max_new_tokens=128):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def batch_generate(self, prompts, max_new_tokens=128):
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_new_tokens=max_new_tokens))
        return results
