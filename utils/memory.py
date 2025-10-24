import json
import re
import torch
from utils.prompts import INSTRUCTION_LST_MEMORY, INSTRUCTION_TOPIC_MEMORY

# Regular expression pattern to extract JSON-like content
pattern = r"{.*?}"

class Memory:
    def __init__(self, type:str):
        self.memory_type=type
        self.short_term_memory = []
        self.long_term_memory = []
        self.topic_memory = []

    def add_to_short_term(self, utterance):
        self.short_term_memory.append(utterance)

    def add_to_long_term(self, utterance):
        self.long_term_memory.append(utterance)

    def add_to_topic_memory(self, entry, raw_dialogue):
        self.topic_memory.append({"entry": entry, "raw_dialogue": raw_dialogue})

    def delete_from_memory(self, utterance):
        if utterance in self.short_term_memory:
            self.short_term_memory.remove(utterance)
        elif utterance in self.long_term_memory:
            self.long_term_memory.remove(utterance)

    def search_memory(self, query, embeddings_model, memory_type='short_term', top_k=5):
        if self.memory_type == 'lst_memory':
            memory_texts = self.short_term_memory+self.long_term_memory
        elif self.memory_type == 'topic_based':
            memory_texts = [memory['entry'] for memory in self.topic_based_memory]
        else:
            raise ValueError("Invalid memory type specified.")

        query_embedding = torch.tensor(embeddings_model.embed_query(query))
        memory_embeddings = torch.tensor(
            embeddings_model.embed_documents([str(mem) for mem in memory_texts])
        )
        similarity_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), memory_embeddings
        ).tolist()

        top_indices = sorted(
            range(len(similarity_scores)),
            key=lambda i: similarity_scores[i],
            reverse=True,
        )[:top_k]

        if memory_type == 'topic_memory':
            return [(self.topic_based_memory[idx]['entry'], self.topic_based_memory[idx]['raw_dialogue']) for idx in top_indices]
        else:
            return [memory_texts[idx] for idx in top_indices]

def convert_session_to_memory(sessions, memory_type, agent, granularity='session'):
    memory_instance = Memory(type=memory_type)
    if granularity == 'session':
        for session in sessions:
            process_session(session, agent, memory_instance)
    elif granularity == 'round':
        process_session(sessions, agent, memory_instance)
    else:
        raise ValueError("Invalid granularity specified.")

    return memory_instance

def process_session(session, agent, memory_instance):
    if memory_instance.memory_type=="lst_memory":
    # Convert session to long-short term memory
        prompt = INSTRUCTION_LST_MEMORY.format(session=session)
    elif memory_instance.memory_type=="topic_memory":
        prompt = INSTRUCTION_TOPIC_MEMORY.format(session=session)
    else:
        prompt=""
        assert "Memory Not Support"
    response = agent(prompt)
    try:
        match = re.search(pattern, response)
        if match:
            extracted_dict = match.group()
            memory_dict = json.loads(extracted_dict)
            if memory_instance.memory_type=="lst_memory":
                for utterance in memory_dict.get("short-term memory", []):
                    memory_instance.add_to_short_term(utterance)
                for utterance in memory_dict.get("long-term memory", []):
                    memory_instance.add_to_long_term(utterance)
            elif memory_instance.memory_type=="topic_memory":
                for entry in memory_dict.get("entries", []):
                    memory_instance.add_to_topic_memory(entry['summary'], entry['raw_dialogue'])
                
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Failed to process response:", response)


# def generate_response(question, memory, metagen_platform):
#     retrieved_memories = memory.search_memory(question, embeddings)
#     prompt = INSTRUCTION_GENERATION.format(memory=retrieved_memories, question=question)
#     response = completion(
#         metagen_platform=metagen_platform,
#         prompt=prompt,
#         model=model_name_or_path,
#     )
#     return response


# def evaluate_responses(responses, gold_answers):
#     bleu_scores = [
#         sentence_bleu([gold_answer.split()], response.split())
#         for response, gold_answer in zip(responses, gold_answers)
#     ]
#     return {
#         "BLEU": sum(bleu_scores) / len(bleu_scores),
#     }


