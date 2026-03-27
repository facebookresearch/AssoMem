# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from utils.prompts import INSTRUCTION_GENERATION
from utils.retrieval import BGERetrieval


class EmbeddingConcatenation:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.retriever = BGERetrieval()
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model = torch.nn.DataParallel(self.model).to('cuda')
        self.generation_model = pipeline("text-generation", model=config.model_name, device_map="balanced")

    def embed_and_concatenate(self, input_query, retrieved_context):
        # Tokenize input query and context
        input_query_tokens = self.tokenizer(input_query, return_tensors='pt').to('cuda')
        context_tokens = self.tokenizer(retrieved_context, return_tensors='pt', padding=True, truncation=True).to('cuda')

        # Get embeddings
        with torch.no_grad():
            input_query_embeddings = self.model(**input_query_tokens).last_hidden_state
            context_embeddings = self.model(**context_tokens).last_hidden_state

        # Concatenate embeddings
        concatenated_embeddings = torch.cat((input_query_embeddings, context_embeddings), dim=1)

        return concatenated_embeddings

    def generate(self):
        data = self.load_dataset()
        results = []

        for sample in data:
            query = sample['question']
            content_sentence = [utterance["content"] for session in sample["haystack_sessions"] for utterance in session]

            # Retrieve top-k similar data samples
            retrieved_context = self.retriever.retrieve(query, content_sentence, top_k=12)

            # Build prompt using the retrieved context
            context_str = " ".join(retrieved_context)
            input_text = INSTRUCTION_GENERATION.format(memory=context_str, question=query)

            # Generate output using the prompt
            try:
                generated = self.generation_model(input_text, max_new_tokens=100, num_return_sequences=1)
                generated_text = generated[0]['generated_text']
            except Exception as e:
                print(f"Error generating text for query: {query}. Error: {e}")
                generated_text = ""

            results.append({"input": query, "output": generated_text})

        return results

    def load_dataset(self):
        try:
            with open(self.config.dataset_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def save_results(self, results):
        try:
            with open(self.config.output_path, 'w') as outfile:
                json.dump(results, outfile, indent=4)
        except Exception as e:
            print(f"Error saving results: {e}")


# if __name__ == "__main__":
#     from config import get_config
#
#     config = get_config()
#     embedding_concat = EmbeddingConcatenation(config)
#     generated_texts = embedding_concat.generate()
#     embedding_concat.save_results(generated_texts)
#     print(f"Generated results saved to {config.output_path}")
