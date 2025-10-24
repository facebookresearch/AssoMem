import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from config import get_config
from utils.prompts import INSTRUCTION_GENERATION
from utils.retrieval import BGERetrieval
from utils.dataset_process import DatasetProcessor

def generate_for_level(query, retrieved_content, model, tokenizer, k_values, level, config):
    results = []
    for k in k_values:
        # Use the top-k retrieved content
        context = " ".join(retrieved_content[:k])
        prompt = INSTRUCTION_GENERATION.format(memory=context, question=query)

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True, padding=True).to(model.device)

        # Split into chunks with overlap
        chunks = []
        for i in range(0, len(input_ids[0]), config.chunk_size - config.overlap):
            end = min(i + config.chunk_size, len(input_ids[0]))
            chunks.append(input_ids[:, i:end])

        # Generate text for each chunk
        generated_texts = []
        for chunk in chunks:
            with torch.no_grad():
                output_ids = model.generate(chunk, max_length=config.max_length, num_return_sequences=1)
            generated_texts.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        # Concatenate outputs
        generated_text = ''.join(generated_texts).replace(prompt, "").strip()

        results.append({
            'query': query,
            'retrieved_content': context,
            'generated_text': generated_text,
            'k': k,
            'level': level
        })
    return results

def generate_responses(data, retriever, model, tokenizer, k_values, config):
    all_results = []
    for sample in tqdm(data, desc="Processing entries"):
        query = sample['question']  # Assuming each sample has a 'question' field
        content_sentence = [utterance["content"] for session in sample["haystack_sessions"] for utterance in session]
        content_session = [str(session) for session in sample["haystack_sessions"]]

        # Retrieve top-k similar data samples
        retrieved_content_sentence = retriever.retrieve(query, content_sentence, top_k=max(k_values))
        retrieved_content_session = retriever.retrieve(query, content_session, top_k=max(k_values))

        # Generate responses for both levels
        all_results.extend(generate_for_level(query, retrieved_content_sentence, model, tokenizer, k_values, 'sentence', config))
        all_results.extend(generate_for_level(query, retrieved_content_session, model, tokenizer, k_values, 'session', config))

    return all_results

if __name__ == "__main__":
    config = get_config()
    data_processor = DatasetProcessor(dataset_name=config.dataset_name, data_path=config.dataset_path)
    data = data_processor.load_dataset()
    retriever = BGERetrieval()

    # Initialize the accelerator
    accelerator = Accelerator()

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_path)

    # Prepare the model for distributed inference
    model = accelerator.prepare(model)

    # Set k values from 1 to 12
    k_values = list(range(1, 13))

    # Generate responses
    results = generate_responses(data[:100], retriever, model, tokenizer, k_values, config)

    # Save results to a JSON file
    with open(os.path.join(config.output_path, 'retrieval.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("Generation complete. Results saved to 'retrieval.json'.")