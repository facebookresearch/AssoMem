import re
pattern = r"{.*}"
import os
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from utils.prompts import INSTRUCTION_LST_MEMORY, INSTRUCTION_TOPIC, INSTRUCTION_GENERATION
from utils.memory import Memory
from config import get_config
from utils.dataset_process import DatasetProcessor

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm

def run_memory_generator(dataset, memory_type: str, generator_path, instruction: str):
    # Initialize the memory object
    memory_all = []

    # prepare accelerator
    accelerator=Accelerator()
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(generator_path)
    model = AutoModelForCausalLM.from_pretrained(generator_path)
    tokenizer.pad_token=tokenizer.eos_token
    
    # Prepare model for distributed inference
    model = accelerator.prepare(model)

    # Iterate over the dataset
    for data in tqdm(dataset, desc="Generating Memory"):
        memory = Memory(memory_type)
        for session in data["haystack_sessions"]:
            # Prepare the input prompt
            input_prompt = instruction.format(session=data)
            
            # Tokenize the input
            inputs = tokenizer(input_prompt, return_tensors="pt")
            
            # Generate a response using the model
            outputs = model.generate(**inputs, max_length=128000, num_return_sequences=1)
            
            # Decode the generated response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
            if memory_type == "LST":
                try:
                    match = re.search(pattern, generated_text)
                    if match:
                        extracted_dict=match.group()
                        memory_dict = json.loads(extracted_dict)
                        for utterance in memory_dict.get("short-term memory", []):
                            memory.add_to_short_term(utterance)
                        for utterance in memory_dict.get("long-term memory", []):
                            memory.add_to_long_term(utterance)
                except Exception as e:
                    print(e)
            elif memory_type == "Topic":
                memory.add_to_topic_memory(topic=generated_text, raw_dialogue=[content["utterance"] for content in session])
            else:
                print("Not supporting memory type")

            print("reach here")
        memory_all.append(memory.topic_memory)
    
    return memory_all


if __name__=="__main__":
    config=get_config()

    data_processor=DatasetProcessor(dataset_name=config.dataset_name, data_path=config.dataset_path)
    dataset=data_processor.load_dataset()
    model_path=config.model_path
    memory_type="Topic"

    memory_ls=run_memory_generator(dataset=dataset, memory_type=memory_type, generator_path=model_path, instruction=INSTRUCTION_TOPIC)
    with open(r"results/topic_memory_m.json", "w", encoding="utf-8") as f:
        json.dump(memory_ls, f)
    
    