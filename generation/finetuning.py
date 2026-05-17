# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def find_linear_modules(model):
    """Find all 4-bit linear modules suitable for LoRA."""
    import bitsandbytes as bnb

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            parts = name.split(".")
            module_names.add(parts[0] if len(parts) == 1 else parts[-1])
    module_names.discard("lm_head")
    return list(module_names)


def load_training_data(data_path):
    """Load and format the denoising QA dataset for finetuning.

    Args:
        data_path: Path to the JSON dataset from DenoisingDatasetBuilder.

    Returns:
        Tuple of (train_dataset, eval_dataset) as HuggingFace Datasets.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = {
        "context": [d["context"] for d in raw_data],
        "question": [d["query"] for d in raw_data],
        "answer": [d["answer"] for d in raw_data],
    }

    dataset = Dataset.from_dict(records)

    # 90/10 train/eval split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def format_for_chat(row, tokenizer, system_instruction):
    """Format a row into a chat template string."""
    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": f"### user query:\n{row['question']}\n### user memory:\n{row['context']}",
        },
        {"role": "assistant", "content": row["answer"]},
    ]
    row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return row


def run_finetuning(config):
    """Run QLoRA multi-task finetuning.

    Args:
        config: Config namespace with model_path, save_path, lora_r,
            lora_alpha, lora_dropout, batch_size, learning_rate, num_epochs,
            and dataset_path.
    """
    # Determine dtype
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA config
    modules = find_linear_modules(model)
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )
    model = get_peft_model(model, peft_config)

    # Load and format data
    train_dataset, eval_dataset = load_training_data(config.dataset_path)

    system_instruction = (
        "You are a helpful assistant that answers user's questions. "
        "First, identify the question type from [knowledge, event, temporal, preference, counterfactual]. "
        "Then, answer the question based on the provided memories.\n"
        "Output in the format: [TYPE: <question_type>] <answer>"
    )

    train_dataset = train_dataset.map(
        lambda row: format_for_chat(row, tokenizer, system_instruction),
        num_proc=4,
    )
    eval_dataset = eval_dataset.map(
        lambda row: format_for_chat(row, tokenizer, system_instruction),
        num_proc=4,
    )

    # Completion-only collator: only compute loss on assistant response
    response_template = tokenizer.encode("\nassistant\n", add_special_tokens=False)
    # Fallback for models without a specific assistant token
    if len(response_template) < 2:
        response_template = "\nassistant\n"

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=config.save_path,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=10,
        optim="paged_adamw_32bit",
        num_train_epochs=config.num_epochs,
        eval_strategy="steps",
        eval_steps=0.1,
        logging_steps=10,
        warmup_steps=50,
        learning_rate=config.learning_rate,
        group_by_length=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )

    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    trainer.model.save_pretrained(config.save_path)
    tokenizer.save_pretrained(config.save_path)
