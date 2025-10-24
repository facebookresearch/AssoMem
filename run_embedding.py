import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from tqdm import tqdm
from transformers import pipeline
from config import get_config
from utils.prompts import INSTRUCTION_GENERATION
from utils.retrieval import BGERetrieval
from utils.dataset_process import DatasetProcessor
from utils.embedding_cat import EmbeddingConcatenation


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    embedding_concat = EmbeddingConcatenation(config)
    generated_texts = embedding_concat.generate()
    embedding_concat.save_results(generated_texts)
    print(f"Generated results saved to {config.output_path}")