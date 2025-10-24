import torch
import json
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from utils.evaluation import calculate_recall, calculate_ndcg
from utils.dataset_process import DatasetProcessor
from utils.retrieval import BGERetrieval
from config import get_config


def top_k_embedding_similarity_retrieval(
    data,
    embeddings_model,
    top_k=11,
    granularity="sentence",
    embedding_cache=None,
):
    if embedding_cache is None:
        embedding_cache = {}

    if granularity == "sentence":
        memory = [
            (
                1 if "has_answer" in utterance.keys() and utterance["has_answer"] else 0,
                utterance["content"]
            )
            for session in data["haystack_sessions"]
            for utterance in session
        ]
    elif granularity == "session":
        memory = [
            (
                1 if "answer" in id else 0,
                str(session)
            )
            for id, session in zip(data["haystack_session_ids"], data["haystack_sessions"])
        ]
    else:
        raise ValueError("Granularity must be either 'sentence' or 'session'.")

    question = data["question"]

    if question not in embedding_cache:
        embedding_cache[question] = torch.tensor(embeddings_model.embed_query(question))
    query_embedding = embedding_cache[question]

    memory_embeddings = []

    with tqdm(total=len(memory)) as pbar:
        for memory_piece in memory:
            if memory_piece[1] not in embedding_cache:
                embedding_cache[memory_piece[1]] = torch.tensor(embeddings_model.embed_content([memory_piece[1]]))[0]
            memory_embeddings.append((memory_piece[0], embedding_cache[memory_piece[1]]))
            pbar.update(1)
    
    memory_embeddings_ = torch.stack([embeddings[1] for embeddings in memory_embeddings])
    similarity_scores = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0), memory_embeddings_
    ).tolist()

    top_indices = sorted(
        range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True
    )
    # top_k_results = [memory[idx] for idx in top_indices]
    
    # relevant_items = [m[1] for m in memory if m[0] == 1]
    # recall = calculate_recall([m[1] for m in top_k_results], relevant_items, top_k)
    # ndcg = calculate_ndcg([m[1] for m in top_k_results], relevant_items, top_k)

    return top_indices, embedding_cache, memory

def retrieval_results(top_k_indices, top_k, memory):
    indices=top_k_indices[:top_k]
    top_k_results = [memory[idx] for idx in indices]
    relevant_items = [m[1] for m in memory if m[0] == 1]
    recall = calculate_recall([m[1] for m in top_k_results], relevant_items, top_k)
    ndcg = calculate_ndcg([m[1] for m in top_k_results], relevant_items, top_k)

    return recall, ndcg, top_k_results


def obtain_results(dataset, embeddings):
    recall_results_sentence = {k: [] for k in range(1, 11)}
    ndcg_results_sentence = {k: [] for k in range(1, 11)}
    recall_results_session = {k: [] for k in range(1, 11)}
    ndcg_results_session = {k: [] for k in range(1, 11)}
    embedding_cache = {}
    top_k_results_all_sentence = []
    top_k_results_all_session = []

    with tqdm(total=len(dataset) * len(recall_results_sentence)) as pbar:
        for data in dataset:
            top_indices_sentence, embedding_cache, memory_sentence = top_k_embedding_similarity_retrieval(
                data, embeddings, granularity="sentence", embedding_cache=embedding_cache
            )
            top_indices_session, embedding_cache, memory_session = top_k_embedding_similarity_retrieval(
                data, embeddings, granularity="session", embedding_cache=embedding_cache
            )
            top_k_results_sentence=[memory_sentence[idx] for idx in top_indices_sentence]
            top_k_results_session=[memory_session[idx] for idx in top_indices_session]
            top_k_results_all_sentence.append(top_k_results_sentence)
            top_k_results_all_session.append(top_k_results_session)
            for k in range(1, 11):
                recall_sentence, ndcg_sentence,top_k_results_sentence=retrieval_results(top_indices_sentence, k, memory_sentence)
                recall_results_sentence[k].append(recall_sentence)
                ndcg_results_sentence[k].append(ndcg_sentence)
                recall_session, ndcg_session, top_k_results_session=retrieval_results(top_indices_session, k, memory_session)
                recall_results_session[k].append(recall_session)
                ndcg_results_session[k].append(ndcg_session)
                pbar.update(1)

    average_recall_sentence_l = {k: sum(recall_results_sentence[k]) / len(recall_results_sentence[k]) for k in recall_results_sentence}
    average_ndcg_sentence_l = {k: sum(ndcg_results_sentence[k]) / len(ndcg_results_sentence[k]) for k in ndcg_results_sentence}
    average_recall_session_l = {k: sum(recall_results_session[k]) / len(recall_results_session[k]) for k in recall_results_session}
    average_ndcg_session_l = {k: sum(ndcg_results_session[k]) / len(ndcg_results_session[k]) for k in ndcg_results_session}

    print("Sentence Granularity:")
    print("{:<10} {:<10} {:<10}".format("k", "Recall", "NDCG"))
    for k in range(1, 11):
        print("{:<10} {:<10.4f} {:<10.4f}".format(k, average_recall_sentence_l[k], average_ndcg_sentence_l[k]))

    print("\nSession Granularity:")
    print("{:<10} {:<10} {:<10}".format("k", "Recall", "NDCG"))
    for k in range(1, 11):
        print("{:<10} {:<10.4f} {:<10.4f}".format(k, average_recall_session_l[k], average_ndcg_session_l[k]))
    
    return embedding_cache, top_k_results_all_sentence, top_k_results_all_session

if __name__ == "__main__":
    config = get_config()
    data_processor = DatasetProcessor(dataset_name=config.dataset_name, data_path=config.dataset_path)
    data = data_processor.load_dataset()
    embeddings = BGERetrieval()

    embedding_cache, top_k_results_sentence, top_k_results_session=obtain_results(dataset=data, embeddings=embeddings)

    with open(r"results/embedding_cache.json", "w", encoding="utf-8") as f:
        json.dump(embedding_cache, f)

    with open(r"results/top_k_results_sentence.json", "w", encoding="utf-8") as f:
        json.dump(top_k_results_sentence, f)
    
    with open(r"results/top_k_retuls_session.json", "w", encoding="utf-8") as f:
        json.dump(top_k_results_session, f)