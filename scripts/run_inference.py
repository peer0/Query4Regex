from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os
import sys
import json
import argparse
import random
from transformers import BitsAndBytesConfig
from copy import deepcopy
from string import Template

def main():
    parser = argparse.ArgumentParser()
    
    # Argument
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--reasoning", action="store_true", default=False)
    parser.add_argument("--dataset-path", default="./data/test.jsonl", type=str)
    parser.add_argument("--fewshot-path", default="./data/train.jsonl", type=str)
    parser.add_argument("--shot", choices=["five", "zero"], default="zero", type=str)
    parser.add_argument("--max-think-tokens", default=4096, type=int)
    parser.add_argument("--max-answer-tokens", default=1024, type=int)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--top-p", default=None, type=float)
    parser.add_argument("--start-idx", default=0, type=int)
    parser.add_argument("--end-idx", default=2147483647, type=int)
    parser.add_argument("--pipeline", choices=["nl2regex", "nl2dsl2regex"], default="nl2regex", type=str)
    parser.add_argument("--instruction-path", default="query4regex/nl/instruction_template.md", type=str)
    parser.add_argument("--instruction-path-dsl", default="query4regex/nl_dsl/instruction_template.md", type=str)
    
    args = parser.parse_args()
    pipeline = args.pipeline

    model_name = args.model_name if args.checkpoint_path is None else args.checkpoint_path
    reasoning = args.reasoning
    dataset_path = args.dataset_path
    fewshot_path = args.fewshot_path
    shot = args.shot
    max_think_tokens = args.max_think_tokens
    max_answer_tokens = args.max_answer_tokens
    do_sample = args.do_sample
    temperature = args.temperature
    top_p = args.top_p
    start_idx = args.start_idx
    end_idx = args.end_idx
    if pipeline == 'nl2dsl2regex':
        instruction_path = args.instruction_path_dsl
    else:
        instruction_path = args.instruction_path
    
    if start_idx >= end_idx:
        print("Start index is larger or equal to End index!")
        exit(1)
        
    # Loading Instruction
    with open(instruction_path, 'r') as f:
        instruction = f.read()
    
    # Loading Dataset
    with open(dataset_path, 'r') as f:
        data = [json.loads(i) for i in f.readlines()]
        
    # Setting Output Path
    path = f"./result/generation/{pipeline}/{shot}-shot/"
    os.makedirs(path, exist_ok=True)
    
    # Loading Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        quantization_config=quantization_config,
        device_map="auto"
    )
    model.eval()

    # Setting Config and Variables for Reasoning
    if reasoning:
        thinking_generation_config = deepcopy(model.generation_config)
        thinking_generation_config.max_new_tokens = max_think_tokens
        think_end_token = "</think>"
        if len(tokenizer.encode(think_end_token, add_special_tokens=False))==1:
            thinking_generation_config.eos_token_id = tokenizer.encode(think_end_token, add_special_tokens=False)[0]
        else:
            thinking_generation_config.stop_strings = think_end_token

    # Setting Config for Sampling
    model.generation_config.do_sample = do_sample
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    if do_sample:
        if temperature != None:
            model.generation_config.temperature = temperature
        if top_p != None:
            model.generation_config.top_p = top_p

    # Output Path
    result_path = path+model_name.split("/")[-1]
    result_path += f"_{reasoning}"
    if do_sample:
        result_path += f"_{model.generation_config.temperature}_{model.generation_config.top_p}"
    if start_idx > 0:
        result_path += f"_{start_idx}"
    result_path += ".jsonl"
    
    # Setting Maximum Answer Tokens
    answer_generation_config = deepcopy(model.generation_config)
    answer_generation_config.max_new_tokens = max_answer_tokens

    # Loading Prompts
    if pipeline == 'nl2dsl2regex':
        prompt_template = Template("Given the following regular expressions:\n${regex_inputs}\n\nInstruction: ${instruction}\n\nOps_dsl: ${ops_dsl}\n\nResulting regex:")
    else:
        prompt_template = Template("Given the following regular expressions:\n${regex_inputs}\n\nInstruction: ${instruction}\n\nResulting regex:")
    
    fewshot_prompt = ""
    if shot == "five":
        with open(fewshot_path, 'r') as f:
            fewshot_data = [json.loads(i) for i in f.readlines()]
        random.seed(42) # for reproducibility
        fewshot_examples = random.sample(fewshot_data, 5)
        for example in fewshot_examples:
            regex_inputs = "\n".join([f"{name}: {regex}" for name, regex in example['inputs'].items()])
            if pipeline == 'nl2dsl2regex':
                fewshot_prompt += prompt_template.substitute(
                    regex_inputs=regex_inputs,
                    instruction=example['instruction'],
                    ops_dsl=example['ops_dsl']
                ) + f" {example['gold_regex']}\n\n"
            else:
                fewshot_prompt += prompt_template.substitute(
                    regex_inputs=regex_inputs,
                    instruction=example['instruction']
                ) + f" {example['gold_regex']}\n\n"

    # Trimming Already Existing Files
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            lines = f.readlines()
        if lines:
            try:
                last_line = json.loads(lines[-1])
                start_idx = max(start_idx, last_line["idx"] + 1)
            except json.JSONDecodeError:
                pass


    # Start Inferencing
    with open(result_path, 'a') as f:
        for i, x in enumerate(tqdm(data[start_idx:end_idx])):
            # Creating Queries
            regex_inputs = "\n".join([f"{name}: {regex}" for name, regex in x['inputs'].items()])
            if pipeline == 'nl2dsl2regex':
                query = fewshot_prompt + prompt_template.substitute(
                    regex_inputs=regex_inputs,
                    instruction=x['instruction'],
                    ops_dsl=x['ops_dsl']
                )
            else:
                query = fewshot_prompt + prompt_template.substitute(
                    regex_inputs=regex_inputs,
                    instruction=x['instruction']
                )
            
            final_query = instruction + "\n\n" + query
            
            # Applying Chat Template
            input_template = tokenizer.apply_chat_template([{"role":"user","content":final_query}], tokenize=False, add_generation_prompt=True)
            
            # Tokenizing
            inputs = tokenizer(input_template, add_special_tokens=False, return_tensors="pt").to('cuda')
            query_length = len(inputs["input_ids"][0])
            
            # Reasoning Inference
            if reasoning:
                if len(tokenizer.encode(think_end_token, add_special_tokens=False))==1:
                    output_ids = model.generate(**inputs, generation_config=thinking_generation_config)[0]
                else:
                    output_ids = model.generate(**inputs, generation_config=thinking_generation_config, tokenizer=tokenizer)[0]
                output_string = tokenizer.decode(output_ids[query_length:])
                
                inputs = tokenizer(input_template+output_string.split(think_end_token)[0]+think_end_token, add_special_tokens=False, return_tensors="pt").to('cuda')
                thinking_length = len(inputs["input_ids"][0]) - query_length

            # Answer Inference
            output_ids = model.generate(**inputs, generation_config=answer_generation_config, tokenizer=tokenizer)[0]
            if reasoning:
                answer_length = len(output_ids) - query_length - thinking_length
            else:
                answer_length = len(output_ids) - query_length
            
            # Resulting Output
            result = {
                "idx": i + start_idx,
                "inputs": x['inputs'],
                "instruction": x['instruction'],
                "op_dsl": x['ops_dsl'],
                "gold_regex": x['gold_regex'],
                "prompt": final_query,
            }
            if reasoning:
                result["generated_thinking"] = tokenizer.decode(output_ids[query_length:query_length+thinking_length])
            result["generated_answer"] = tokenizer.decode(output_ids[-answer_length:])

            # Saving Output
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
            
if __name__ == "__main__":
    main()
