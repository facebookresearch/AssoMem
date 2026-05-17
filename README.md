# AssoMem

Source implementation and dataset for the paper **"AssoMem: Scalable Memory QA with Multi-Signal Associative Retrieval"** ([arXiv:2510.10397](https://arxiv.org/abs/2510.10397)).

AssoMem is a memory-augmented QA system that organizes long-horizon dialogue history into an associative memory graph and retrieves relevant memories using a three-signal ranking framework (Relevance, Importance, Temporal).

## Overview

The pipeline consists of three stages:

### 1. Graph Construction (Offline)

- **Clue Tagging**: An LLM summarizes each dialogue session into a concise topic (clue).
- **Clue Merging**: Clues with cosine similarity above a threshold are merged, grouping their sessions and utterances.
- **Associative Graph**: A bipartite graph is built with clue nodes and utterance nodes, connected by ownership edges and utterance-utterance similarity edges.

### 2. RIT Scoring (Runtime)

Each candidate utterance is scored by three signals:

- **Relevance (R)**: Cosine similarity between query and utterance embeddings.
- **Importance (I)**: Personalized PageRank on the associative graph, with teleportation biased toward query-similar utterances.
- **Temporal (T)**: Multi-scale exponential decay based on utterance recency.

Signal weights are assigned per query type using Conditional Mutual Information (CMI) with softmax temperature scaling.

### 3. Retrieval + Generation

- **Two-step Retrieval**: First retrieve top-K clues, then rank all utterances under those clues using the combined RIT score.
- **Answer Generation**: An LLM generates answers conditioned on the retrieved memory context.
- **Multi-task Fine-tuning**: QLoRA fine-tuning with joint question type prediction and answer generation on a denoising dataset mixing golden and noisy memories.

## Project Structure

```
AssoMem/
├── run.py                        # Main entry point
├── config.py                     # All configuration parameters
├── requirements.txt              # Dependencies (open-source only)
│
├── graph/                        # Stage 1: Graph Construction
│   ├── clue_tagger.py            # LLM-based clue extraction per session
│   ├── clue_merger.py            # Pairwise cosine merging with threshold δ
│   └── associative_graph.py      # Bipartite networkx graph
│
├── scoring/                      # Stage 2: RIT Scoring
│   ├── relevance.py              # Cosine similarity scoring
│   ├── importance.py             # Personalized PageRank
│   ├── temporal.py               # Exponential decay scoring
│   ├── fusion.py                 # CMI-based adaptive weight assignment
│   └── rit_ranker.py             # Combined RIT ranking
│
├── retrieval/                    # Stage 3a: Retrieval
│   ├── embedding_model.py        # BGE embedding wrapper
│   └── candidate_retrieval.py    # Two-step hybrid retrieval
│
├── generation/                   # Stage 3b: Generation & Fine-tuning
│   ├── answer_generator.py       # LLM answer generation
│   ├── dataset_builder.py        # Denoising QA dataset construction
│   └── finetuning.py             # QLoRA multi-task fine-tuning
│
├── evaluation/                   # Evaluation
│   ├── retrieval_metrics.py      # Recall@k, NDCG@k
│   ├── generation_metrics.py     # BERTScore, LLM-as-Judge
│   └── run_eval.py               # Evaluation entry point
│
├── baselines/                    # Baseline methods
│   ├── icl_baseline.py           # In-context learning baseline
│   └── rag_baseline.py           # Flat RAG retrieval baseline
│
├── utils/                        # Shared utilities
│   ├── prompts.py                # Prompt templates
│   ├── data_loader.py            # Dataset loading helpers
│   └── llm_client.py             # HuggingFace LLM inference client
│
├── scripts/                      # SLURM job scripts
│   ├── build_graph.sh
│   ├── run_retrieval.sh
│   ├── run_finetuning.sh
│   └── run_evaluation.sh
│
└── dataset/                      # Included datasets
    └── MeetingQA/
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies (all open-source):
- `torch >= 2.0`
- `transformers >= 4.36`
- `networkx >= 3.0` (graph construction and PageRank)
- `FlagEmbedding >= 1.2` (BGE embeddings)
- `peft >= 0.7`, `bitsandbytes >= 0.41`, `trl >= 0.7` (QLoRA fine-tuning)
- `bert-score >= 0.3` (evaluation)

## Usage

### Full Pipeline

Run the entire pipeline end-to-end (graph construction, retrieval, generation, evaluation):

```bash
python run.py --stage pipeline \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_name LongMemEval \
    --dataset_path datasets/longmemeval_m.json \
    --embedding_model BAAI/bge-large-en-v1.5 \
    --output_path results
```

### Individual Stages

Each stage can be run independently:

```bash
# Stage 1: Build the associative memory graph
python run.py --stage graph \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path datasets/longmemeval_m.json \
    --clue_merge_threshold 0.65 \
    --utterance_sim_threshold 0.75 \
    --graph_save_path results/graph

# Stage 2: Retrieve memories for all queries
python run.py --stage retrieve \
    --dataset_path datasets/longmemeval_m.json \
    --graph_save_path results/graph \
    --top_k_clues 10 \
    --top_k_utterances 6

# Stage 3: Generate answers from retrieved context
python run.py --stage generate \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --output_path results

# Stage 4: Fine-tune with QLoRA on denoising dataset
python run.py --stage finetune \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path datasets/curated_data.json \
    --save_path models/fine-tuned/assomem \
    --lora_r 16 --lora_alpha 32 --num_epochs 3

# Stage 5: Evaluate retrieval and generation
python run.py --stage evaluate \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path datasets/longmemeval_m.json \
    --graph_save_path results/graph \
    --output_path results
```

### Baselines

```bash
# In-context learning baseline (all sessions as context)
python -m baselines.icl_baseline \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path datasets/longmemeval_m.json

# Flat RAG baseline (cosine similarity retrieval, no graph)
python -m baselines.rag_baseline \
    --model_path meta-llama/Llama-3.3-70B-Instruct \
    --dataset_path datasets/longmemeval_m.json
```

### SLURM

For cluster environments, use the provided SLURM scripts:

```bash
sbatch scripts/build_graph.sh
sbatch scripts/run_retrieval.sh
sbatch scripts/run_finetuning.sh
sbatch scripts/run_evaluation.sh
```

## Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `--clue_merge_threshold` | 0.65 | Cosine similarity threshold for merging clues |
| `--utterance_sim_threshold` | 0.75 | Threshold for utterance-utterance similarity edges |
| `--ppr_damping` | 0.85 | Damping factor for Personalized PageRank |
| `--temporal_decay_weights` | 3.0 90.0 365.0 | Exponential decay constants (short/mid/long-term) |
| `--cmi_temperature` | 1.0 | Temperature for CMI softmax weight assignment |
| `--top_k_clues` | 10 | Number of clues retrieved in first stage |
| `--top_k_utterances` | 6 | Number of utterances returned after RIT ranking |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--embedding_model` | BAAI/bge-large-en-v1.5 | Embedding model for retrieval |

Run `python run.py --help` for the complete list of parameters.

## Datasets

- **LongMemEval**: Long-horizon memory evaluation benchmark (small/medium/large variants).
- **MeetingQA**: Meeting transcript QA dataset (included in `dataset/MeetingQA/`).

## Citation

```bibtex
@inproceedings{
zhang2026assomem,
title={AssoMem: Scalable Memory {QA} with Multi-Signal Associative Retrieval},
author={Kai Zhang and Xinyuan Zhang and Ejaz Ahmed and Hongda Jiang and Caleb Kumar and Kai Sun and Zhaojiang Lin and Sanat Sharma and Shereen Oraby and AARON COLAK and Ahmed A Aly and Anuj Kumar and Xiaozhong Liu and Xin Luna Dong},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=ZCjWUBwCwE}
}
```

## License

AssoMem is CC-by-NC licensed, as found in the LICENSE file.
