#!/usr/bin/env python3
"""
Experiment runner for the NL-DSL gap extension.
Combines: constrained decoding + iterative verification + multi-feedback levels.

Usage:
    python scripts/run_inference_verified.py \
        --model-name "microsoft/Phi-4" \
        --dataset-path "./data/test.jsonl" \
        --pipeline "nl2regex" \
        --shot "zero" \
        --feedback-level "counterexample" \
        --max-rounds 5 \
        --constrained
"""
import argparse
import json
import os
import random
import sys
from copy import deepcopy
from string import Template

try:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    pass  # Allow --help without GPU dependencies

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.config import DEFAULT_ALPHABET
from query4regex.verify.feedback import FeedbackLevel
from query4regex.verify.loop import VerificationResult, run_verification_loop


class HFModel:
    """Wraps a HuggingFace model to match the GenerativeModel protocol."""

    def __init__(self, model, tokenizer, generation_config, reasoning=False,
                 thinking_generation_config=None, think_end_token="</think>",
                 constrained_generator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.reasoning = reasoning
        self.thinking_generation_config = thinking_generation_config
        self.think_end_token = think_end_token
        self.constrained_generator = constrained_generator

    def generate(self, prompt: str) -> str:
        input_template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.constrained_generator is not None:
            result = self.constrained_generator(input_template)
            return result

        inputs = self.tokenizer(
            input_template, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")
        query_length = len(inputs["input_ids"][0])

        if self.reasoning and self.thinking_generation_config is not None:
            think_cfg = self.thinking_generation_config
            if len(self.tokenizer.encode(self.think_end_token, add_special_tokens=False)) == 1:
                output_ids = self.model.generate(**inputs, generation_config=think_cfg)[0]
            else:
                output_ids = self.model.generate(
                    **inputs, generation_config=think_cfg, tokenizer=self.tokenizer
                )[0]
            output_string = self.tokenizer.decode(output_ids[query_length:])
            full_input = input_template + output_string.split(self.think_end_token)[0] + self.think_end_token
            inputs = self.tokenizer(
                full_input, add_special_tokens=False, return_tensors="pt"
            ).to("cuda")

        output_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            tokenizer=self.tokenizer,
        )[0]
        return self.tokenizer.decode(output_ids[query_length:])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--reasoning", action="store_true", default=False)
    parser.add_argument("--dataset-path", default="./data/test.jsonl", type=str)
    parser.add_argument("--fewshot-path", default="./data/train.jsonl", type=str)
    parser.add_argument("--shot", choices=["five", "zero"], default="zero", type=str)
    parser.add_argument("--pipeline", choices=["nl2regex", "nl2dsl2regex"], default="nl2regex", type=str)
    parser.add_argument("--feedback-level", choices=["binary", "counterexample", "diagnostic"], default="counterexample", type=str)
    parser.add_argument("--max-rounds", default=5, type=int)
    parser.add_argument("--constrained", action="store_true", default=False)
    parser.add_argument("--max-think-tokens", default=4096, type=int)
    parser.add_argument("--max-answer-tokens", default=1024, type=int)
    parser.add_argument("--start-idx", default=0, type=int)
    parser.add_argument("--end-idx", default=2147483647, type=int)
    parser.add_argument("--instruction-path", default="query4regex/nl/instruction_template.md", type=str)
    parser.add_argument("--instruction-path-dsl", default="query4regex/nl_dsl/instruction_template.md", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    feedback_map = {
        "binary": FeedbackLevel.BINARY,
        "counterexample": FeedbackLevel.COUNTEREXAMPLE,
        "diagnostic": FeedbackLevel.DIAGNOSTIC,
    }
    feedback_level = feedback_map[args.feedback_level]

    model_name = args.model_name if args.checkpoint_path is None else args.checkpoint_path
    instruction_path = args.instruction_path_dsl if args.pipeline == "nl2dsl2regex" else args.instruction_path

    with open(instruction_path, "r") as f:
        instruction_text = f.read()

    with open(args.dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    constrained_tag = "_constrained" if args.constrained else ""
    path = f"./result/verified/{args.pipeline}/{args.shot}-shot/{args.feedback_level}_r{args.max_rounds}{constrained_tag}/"
    os.makedirs(path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", quantization_config=quant_config, device_map="auto"
    )
    model.eval()

    gen_config = deepcopy(model.generation_config)
    gen_config.max_new_tokens = args.max_answer_tokens
    gen_config.do_sample = False
    gen_config.pad_token_id = tokenizer.eos_token_id

    thinking_gen_config = None
    if args.reasoning:
        thinking_gen_config = deepcopy(model.generation_config)
        thinking_gen_config.max_new_tokens = args.max_think_tokens
        think_end_token = "</think>"
        if len(tokenizer.encode(think_end_token, add_special_tokens=False)) == 1:
            thinking_gen_config.eos_token_id = tokenizer.encode(think_end_token, add_special_tokens=False)[0]
        else:
            thinking_gen_config.stop_strings = think_end_token

    constrained_generator = None
    if args.constrained:
        from query4regex.constrained.generate import create_constrained_generator
        constrained_generator = create_constrained_generator(model, tokenizer, DEFAULT_ALPHABET)

    hf_model = HFModel(
        model=model,
        tokenizer=tokenizer,
        generation_config=gen_config,
        reasoning=args.reasoning,
        thinking_generation_config=thinking_gen_config,
        constrained_generator=constrained_generator,
    )

    fewshot_prompt = ""
    if args.shot == "five":
        with open(args.fewshot_path, "r") as f:
            fewshot_data = [json.loads(line) for line in f]
        random.seed(42)
        examples = random.sample(fewshot_data, 5)
        prompt_template = Template(
            "Given the following regular expressions:\n${regex_inputs}\n\n"
            "Instruction: ${instruction}\n\nResulting regex:"
        )
        for ex in examples:
            regex_inputs = "\n".join(f"{k}: {v}" for k, v in ex["inputs"].items())
            fewshot_prompt += prompt_template.substitute(
                regex_inputs=regex_inputs, instruction=ex["instruction"]
            ) + f" {ex['gold_regex']}\n\n"

    result_path = os.path.join(path, model_name.split("/")[-1] + ".jsonl")
    start_idx = args.start_idx
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            lines = f.readlines()
        if lines:
            try:
                last = json.loads(lines[-1])
                start_idx = max(start_idx, last["idx"] + 1)
            except json.JSONDecodeError:
                pass

    with open(result_path, "a") as f:
        for i, x in enumerate(tqdm(data[start_idx : args.end_idx])):
            full_instruction = instruction_text + "\n\n" + x["instruction"]
            if fewshot_prompt:
                full_instruction = instruction_text + "\n\n" + fewshot_prompt + x["instruction"]

            vresult = run_verification_loop(
                model=hf_model,
                gold_regex=x["gold_regex"],
                inputs=x["inputs"],
                instruction=full_instruction,
                ops_dsl=x["ops_dsl"],
                pipeline=args.pipeline,
                alphabet=DEFAULT_ALPHABET,
                feedback_level=feedback_level,
                max_rounds=args.max_rounds,
            )

            record = {
                "idx": i + start_idx,
                "inputs": x["inputs"],
                "instruction": x["instruction"],
                "ops_dsl": x["ops_dsl"],
                "gold_regex": x["gold_regex"],
                "success": vresult.success,
                "solved_at_round": vresult.solved_at_round,
                "num_rounds": len(vresult.rounds),
                "rounds": [
                    {
                        "round": r.round_num,
                        "predicted_regex": r.predicted_regex,
                        "parsable": r.parsable,
                        "equivalent": r.equivalent,
                        "feedback": r.feedback,
                    }
                    for r in vresult.rounds
                ],
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
