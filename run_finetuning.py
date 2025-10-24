# base_model = "models/Qwen2.5-32B"
# adapter_model = "models/fine-tuned/Qwen2.5-32B-mem-recall-qa"
data_path = "datasets/longmemeval_data/curated_data.json"
annotation_path = "retrieval.json"

instruction = '''
### TASK DESCRIPTION
You are a helpful assistant that answers user's questions. In this sense, you will have access to user's memory records which contain user's historical informaion.
Please note you will need to identify if the memories are useful or not for you to answer the query. 
If the memories are useful then answer the question based on the memories, otherwise answer the question based on your knowledge or answer "IDK".

### OUTPUT REQUIREMENT
Output the answer to the question only. No matter you use the memory or not, please only output the answer and nothing else.
'''

import wandb

wandb.init(project="meta_memory_recall_finetuning")
import torch
import json
from datasets import Dataset
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from config import get_config
configs=get_config()
adapter_model=configs.save_path

if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    configs.model_path,
    quantization_config=bnb_config,
    device_map="auto",
    # device_map={'': torch.cuda.current_device()},
    # attn_implementation=attn_implementation,
)

tokenizer = AutoTokenizer.from_pretrained(
    configs.model_path, 
    trust_remote_code=True, 
    padding=True, 
    truncation="max_length", 
    return_tensor="pt")

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

modules = find_all_linear_names(model)

print(modules)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model = get_peft_model(model, peft_config)

def format_chat_template(row):
    row_json = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": "### user query:\n"
            + row["question"]
            + "\n"
            + "### user memory:\n"
            + row["context"],
        },
        {"role": "assistant", "content": row["answer"]},
    ]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

with open(data_path, "r", encoding="utf8") as f:
    dataraw = json.load(f)
# with open(annotation_path, "r", encoding="utf8") as f:
#     dataann = json.load(f)

data = {}
data["train"] = {"context": [], "question": [], "answer": []}
data["eval"] = {"context": [], "question": [], "answer": []}
maxlen = 0
for i in range(len(dataraw)):
    split = "train" if i < 1800 else "eval"
    data[split]["question"] += [str(dataraw[i]["query"])]
    data[split]["answer"] += [str(dataraw[i]["answer"])]
    data[split]["context"] += ["Memory evidence:"+"\nMemory evidence:".join(dataraw[i]["memory"])]

dataset = Dataset.from_dict(data["train"])
        
dataset = Dataset.from_dict(data["train"])

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

dataseteval = Dataset.from_dict(data["eval"])

dataseteval = dataseteval.map(
    format_chat_template,
    num_proc=4,
)

training_arguments = SFTConfig(
    output_dir=configs.save_path,
    per_device_train_batch_size=configs.batch_size,
    per_device_eval_batch_size=configs.batch_size,
    gradient_accumulation_steps=10,
    optim="paged_adamw_32bit",
    num_train_epochs=configs.num_epochs,
    eval_strategy="steps",
    eval_steps=0.1,
    logging_steps=10,
    warmup_steps=50,
    logging_strategy="steps",
    learning_rate=configs.learning_rate,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb",
)

instruction_template = "<|start_header_id|>system<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataseteval,
    peft_config=peft_config,
    # max_seq_length=8192,
    # dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    # packing=False,
    data_collator=collator,
)

model.config.use_cache = False
trainer.train()

model.config.use_cache = True

trainer.model.save_pretrained(adapter_model)
