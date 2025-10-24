import argparse

def get_config():
    parser = argparse.ArgumentParser(description="In-Context Learning Configuration")

    # Mode selection
    parser.add_argument('--method_type', type=str, default="context", help="method types: [context, embedding, parameters]")

    # Model and Dataset
    parser.add_argument('--model_name', type=str, default='Llama-3.3-70B-Instruct', help='Name of the model')
    parser.add_argument('--model_path', type=str, default='/models/Llama-3.3-70B-Instruct', help='Path of the model')
    parser.add_argument('--dataset_name', type=str, default='LongMemEval', help='Name of the dataset to process')
    parser.add_argument('--dataset_path', type=str, help='Path to the processed dataset')
    parser.add_argument('--retrieval_name', type=str, default='BAAI/bge-base-en-v1.5', help='Name of the retriever used for retrieval')

    # Training Arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='models/fine-tuned/Qwen2.5-32B-mem-recall-qa')

    # Genertaion/Inference Arguments
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--overlap', type=int, default=50)

    # Granularity
    parser.add_argument('--granularity', type=str, default='utterance', choices=['utterance', 'session', 'round'], help='Granularity of context')

    # Output
    parser.add_argument('--output_path', type=str, default='results', help='Path to save the generated results')

    return parser.parse_args()