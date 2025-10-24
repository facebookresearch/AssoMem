import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel
from bert_score import score as bert_score

# eval for generation
class LLMJudge:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = torch.nn.DataParallel(self.model).to('cuda')

    def judge(self, generated_responses, golden_answers):
        win_count = 0
        total = len(generated_responses)

        for generated, golden in zip(generated_responses, golden_answers):
            prompt = f"Which response is better?\nGenerated: {generated}\nGolden: {golden}\nAnswer:"

            inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
            outputs = self.model.module.generate(**inputs, max_length=50)
            judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Generated" in judgment:
                win_count += 1

        win_rate = win_count / total
        return win_rate

class BertScoreEvaluator:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model = torch.nn.DataParallel(self.model).to('cuda')

    def evaluate(self, generated_responses, golden_answers):
        P, R, F1 = bert_score(generated_responses, golden_answers, model_type='bert-base-uncased', lang='en', device='cuda')
        return P.mean().item(), R.mean().item(), F1.mean().item()

# eval for retrieval
def calculate_recall(predictions, relevant_items, top_k):
    retrieved_relevant = len([p for p in predictions[:top_k] if p in relevant_items])
    total_relevant = len(relevant_items)
    recall = retrieved_relevant / total_relevant if total_relevant else 0
    return recall

def calculate_ndcg(predictions, relevant_items, top_k):
    dcg = sum(1.0 / torch.log2(torch.tensor(idx + 2, dtype=torch.float)) for idx, p in enumerate(predictions[:top_k]) if p in relevant_items)
    idcg = sum(1.0 / torch.log2(torch.tensor(idx + 2, dtype=torch.float)) for idx in range(min(len(relevant_items), top_k)))
    ndcg = dcg / idcg if idcg else 0
    return ndcg


def load_data(generated_path, golden_path):
    with open(generated_path, 'r') as gen_file:
        generated_data = json.load(gen_file)

    with open(golden_path, 'r') as gold_file:
        golden_data = json.load(gold_file)

    generated_responses = [entry['output'] for entry in generated_data]
    golden_answers = [entry['golden'] for entry in golden_data]

    return generated_responses, golden_answers

if __name__ == "__main__":
    # Paths to the generated responses and golden answers
    generated_path = 'generated_results.json'
    golden_path = 'golden_answers.json'

    # Load data
    generated_responses, golden_answers = load_data(generated_path, golden_path)

    # Initialize evaluators
    llm_judge = LLMJudge('LlaMa-3.3-70b-Instruct')
    bert_evaluator = BertScoreEvaluator()

    # Evaluate using LLM-as-a-Judge
    win_rate = llm_judge.judge(generated_responses, golden_answers)
    print(f"Win Rate (LLM-as-a-Judge): {win_rate:.2f}")

    # Evaluate using BERTScore
    P, R, F1 = bert_evaluator.evaluate(generated_responses, golden_answers)
    print(f"BERTScore - Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")