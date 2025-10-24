import json
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from prompts import Prompts

from retrieval import BGERetrieval

class EmbeddingConcatenation:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.retriever=BGERetrieval()
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model = torch.nn.DataParallel(self.model).to('cuda')
        self.generation_model = pipeline("text-generation", model=config.model_name, device_map="balanced")
        self.prompts = Prompts()

    def embed_and_concatenate(self, input_query, retrieved_context):
        # Tokenize input query and context
        input_query_tokens = self.tokenizer(input_query, return_tensors='pt').to('cuda')
        context_tokens = self.tokenizer(retrieved_context, return_tensors='pt').to('cuda')

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
            prompt_text = self.prompts.get_prompt("INSTRUCTION_GENERATION")
            query = sample['question']  # Assuming each sample has a 'question' field
            content_sentence = [utterance["content"] for session in sample["haystack_sessions"] for utterance in session]
            content_session = [str(session) for session in sample["haystack_sessions"]]

            # Retrieve top-k similar data samples
            retrieved_context = self.retriever.retrieve(query, content_sentence, top_k=max(list(range(1, 13))))
            # retrieved_content_session = self.retriever.retrieve(query, content_session, top_k=max(list(range(1, 13))))
            concatenated_embeddings = self.embed_and_concatenate(sample.strip(), retrieved_context)

            # Generate output using concatenated embeddings
            try:
                # Convert embeddings back to text input for generation
                input_ids = torch.argmax(concatenated_embeddings, dim=-1)
                input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

                generated = self.generation_model(input_text, max_length=100, num_return_sequences=1)
                generated_text = generated[0]['generated_text']
            except Exception as e:
                print(f"Error generating text for sample: {sample.strip()}. Error: {e}")
                generated_text = ""

            results.append({"input": sample.strip(), "output": generated_text})

        return results

    def load_dataset(self):
        try:
            with open(self.config.dataset_path, 'r') as file:
                data = file.readlines()  # or json.load(file) for JSON
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

#     config = get_config()
#     embedding_concat = EmbeddingConcatenation(config)
#     generated_texts = embedding_concat.generate()
#     embedding_concat.save_results(generated_texts)
#     print(f"Generated results saved to {config.output_path}")