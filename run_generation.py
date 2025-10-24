import json
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config import get_config
from utils.prompts import INSTRUCTION_GENERATION

def load_generation_pipeline(model_path):
    """
    Loads a text generation pipeline with model parallelism.
    Args:
        model_path (str): Path to the pretrained model directory.
    Returns:
        generator (transformers.Pipeline): Text generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",         # <--- This shards the model across all GPUs
        torch_dtype="auto"         # <--- Use appropriate dtype (fp16/bf16)
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
        # No device argument needed; model is already sharded
    )
    return generator

def generate_text(generator, input_text, max_new_tokens=80):
    tokenizer = generator.tokenizer
    input_ids = tokenizer.encode(input_text, truncation=True, padding=True, max_length=4096)
    truncated_input = tokenizer.decode(input_ids)
    result = generator(
        truncated_input,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return result[0]['generated_text']

if __name__ == "__main__":
    config = get_config()
    model_path = config.model_path
    dataset_path = config.dataset_path

    with open(dataset_path, "r", encoding="utf8") as f:
        dataset = json.load(f)

    # Only one pipeline/model, sharded across all GPUs
    generator = load_generation_pipeline(model_path)

    generation_results = []
    for data in tqdm(dataset, total=len(dataset), desc="Generating:"):
        memory = "Memory evidence:".join(data["memory"][:6])
        input_text = INSTRUCTION_GENERATION.format(question=data["question"], memory=memory)
        output = generate_text(generator, input_text, max_new_tokens=128)
        generation_results.append(output)

    result_path = f"results/generation_result_{config.model_name}"
    with open(result_path, "w", encoding="utf8") as f:
        json.dump(generation_results, f, ensure_ascii=False, indent=2)