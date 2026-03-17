import json
import re
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query4regex.eval.custom_equivalence import are_equivalent

def extract_generated_regex(generated_answer: str) -> str:
    """
    Extracts the regex from the oxed{...} format.
    """
    match = re.search(r'oxed\{(.*)\}', generated_answer)
    if match:
        return match.group(1)
    return ""

def main():
    # Example file, you can change this to any of your result files
    # result_file = 'result/generation/zero-shot/gpt-oss-20b_False.jsonl'
    result_file = 'result/generation/five-shot/DeepSeek-R1-Distill-Llama-8B_False.jsonl'

    try:
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                gold_regex = data['gold_regex']
                generated_answer = data['generated_answer']

                generated_regex = extract_generated_regex(generated_answer)

                print(f"Gold Regex:      {gold_regex}")
                print(f"Generated Regex: {generated_regex}")

                # 1. String exact match
                exact_match = (gold_regex == generated_regex)
                print(f"Exact Match:     {exact_match}")

                # 2. Regex equivalence check (if not an exact match)
                if not exact_match:
                    equivalent = are_equivalent(gold_regex, generated_regex)
                    print(f"Equivalent:      {equivalent}")
                
                print("---")

    except FileNotFoundError:
        print(f"Error: Result file not found at '{result_file}'")
        print("Please make sure the path is correct.")

if __name__ == "__main__":
    main()
