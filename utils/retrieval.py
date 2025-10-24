import os

import torch
from FlagEmbedding import FlagAutoModel
from torch.nn.functional import cosine_similarity

class BGERetrieval:
    def __init__(self, retriever_name='BAAI/bge-large-en-v1.5', use_fp16=True):
        self.model = FlagAutoModel.from_finetuned(
            retriever_name,
            query_instruction_for_retrieval="Given a question, you need to retrieve relevant information for answering the question.",
            use_fp16=use_fp16
        )

    def embed_query(self, query):
        query_embeddings=self.model.encode_queries(query)
        return query_embeddings

    def embed_content(self, content):
        content_embeddings=self.model.encode(content)
        return content_embeddings

    def retrieve(self, query, content, top_k=10):
        query_embeddings = torch.tensor(self.embed_query(query))[0]
        content_embeddings = torch.tensor(self.embed_content(content))[0]

        # print(query_embeddings)
        # print(content_embeddings)

        similarities = torch.nn.functional.cosine_similarity(query_embeddings.unsqueeze(0), content_embeddings.unsqueeze(0)).tolist()
        print(similarities)
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

        return [content[idx] for idx in top_indices]

    def hybrid_retrieval(self, query, content, top_k=10):
        # Step 1: Embed query and all topics
        query_embedding = torch.tensor(self.embed_query(query)).squeeze(0)
        topics = [item['topic'] for item in content]
        topic_embeddings = torch.tensor(self.embed_content(topics))
        
        # Step 2: Retrieve top-k topics
        topic_similarities = cosine_similarity(query_embedding.unsqueeze(0), topic_embeddings).squeeze(0)
        top_topic_indices = topic_similarities.topk(top_k).indices.tolist()
        top_topic_groups = [content[idx] for idx in top_topic_indices]
        
        # Step 3: For each top topic, retrieve top-k utterances
        candidate_utterances = []
        for group in top_topic_groups:
            utterances = group['utterances']
            utterance_embeddings = torch.tensor(self.embed_content(utterances))
            utterance_similarities = cosine_similarity(query_embedding.unsqueeze(0), utterance_embeddings).squeeze(0)
            top_utt_indices = utterance_similarities.topk(top_k).indices.tolist()
            for idx in top_utt_indices:
                candidate_utterances.append({
                    "utterance": utterances[idx],
                    "similarity": utterance_similarities[idx].item()
                })
        
        # Step 4: Rerank all candidate utterances and return top-k utterances
        candidate_utterances = sorted(candidate_utterances, key=lambda x: x['similarity'], reverse=True)
        return [item['utterance'] for item in candidate_utterances[:top_k]]




# # Example usage
# if __name__ == "__main__":
#     # Initialize the retrieval system
#     retrieval_system = BGERetrieval()

#     # Example data
#     data_entries = [
#         "The weather is nice today.",
#         "I love programming in Python.",
#         "Let's go for a walk.",
#         "Machine learning is fascinating.",
#         "How about a game of chess?"
#     ]

#     # User query
#     query = "I enjoy coding."

#     # Retrieve top-k similar entries
#     top_entries = retrieval_system.retrieve(query, data_entries, top_k=3)
#     print("Top similar entries:", top_entries)