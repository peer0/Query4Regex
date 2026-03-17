import json
import re
import os
import glob
import sys
from collections import defaultdict
import pandas as pd

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query4regex.eval.custom_equivalence import are_equivalent

CUSTOM_OPERATORS = ['&', '~', '{']

def extract_generated_regex(generated_answer: str) -> str:
    match = re.search(r'oxed\{(.*)\}', generated_answer, re.DOTALL)
    return match.group(1) if match else ""

def count_operators(regex_str: str) -> int:
    return sum(regex_str.count(op) for op in CUSTOM_OPERATORS)

def analyze_file(filepath):
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    total_cases = len(results)
    if total_cases == 0:
        return None, None

    parsing_success_count = 0
    total_score = 0
    op_counts = defaultdict(lambda: {'count': 0, 'correct': 0})
    detailed_results = []

    for data in results:
        gold_regex = data.get('gold_regex', '')
        generated_answer = data.get('generated_answer', '')
        generated_regex = extract_generated_regex(generated_answer)
        
        equivalence_result = are_equivalent(gold_regex, generated_regex)
        is_parsable = equivalence_result is not None
        score = 1 if equivalence_result is True else 0

        if is_parsable:
            parsing_success_count += 1
        total_score += score
        
        num_ops = count_operators(gold_regex)
        op_counts[num_ops]['count'] += 1
        op_counts[num_ops]['correct'] += score

        detailed_results.append({
            'gold_regex': gold_regex,
            'generated_regex': generated_regex,
            'exact_match': gold_regex == generated_regex,
            'equivalent': equivalence_result is True,
            'parsable': is_parsable
        })

    # Calculate scores
    parsing_rate = (parsing_success_count / total_cases) * 100
    avg_score = (total_score / total_cases) * 100

    op_scores = {}
    for op_num in [1, 2, 3]:
        count = op_counts[op_num]['count']
        correct = op_counts[op_num]['correct']
        op_scores[op_num] = (correct / count) * 100 if count > 0 else 0

    analysis_summary = {
        'parsing_rate': parsing_rate,
        'average_score': avg_score,
        'op_1_score': op_scores.get(1, 0),
        'op_2_score': op_scores.get(2, 0),
        'op_3_score': op_scores.get(3, 0),
    }

    return analysis_summary, detailed_results

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generation_dir = os.path.join(base_dir, 'result', 'generation')
    evaluation_dir = os.path.join(base_dir, 'result', 'evaluation')
    
    search_patterns = [
        os.path.join(generation_dir, 'nl2regex', '**', '*.jsonl'),
        os.path.join(generation_dir, 'nl2dsl2regex', '**', '*.jsonl')
    ]

    all_files = []
    for pattern in search_patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    if not all_files:
        print("No result files found.")
        return

    all_model_results = []

    for filepath in all_files:
        print(filepath)
        analysis, detailed_results = analyze_file(filepath)
        if not analysis:
            continue

        # Extract metadata from filepath
        parts = filepath.split(os.sep)
        try:
            approach = parts[-4]
            shot = parts[-3]
            filename = parts[-1]
            model_name_part = filename.replace('.jsonl', '')
            is_reasoning = model_name_part.endswith('_True')
            model_name = model_name_part.replace('_True', '').replace('_False', '')
        except IndexError:
            continue

        # Save detailed evaluation file
        eval_output_dir = os.path.join(evaluation_dir, approach, shot)
        os.makedirs(eval_output_dir, exist_ok=True)
        eval_output_path = os.path.join(eval_output_dir, filename)
        with open(eval_output_path, 'w') as f:
            for res in detailed_results:
                f.write(json.dumps(res) + '\n')

        all_model_results.append({
            'Approach': approach,
            'Shot': shot,
            'Model': model_name,
            'Reasoning': is_reasoning,
            'Parsing Rate (%)': f"{analysis['parsing_rate']:.2f}",
            'Avg Score (%)': f"{analysis['average_score']:.2f}",
            'Ops=1 Score (%)': f"{analysis['op_1_score']:.2f}",
            'Ops=2 Score (%)': f"{analysis['op_2_score']:.2f}",
            'Ops=3 Score (%)': f"{analysis['op_3_score']:.2f}",
        })

    # Create a pandas DataFrame for pretty printing
    df = pd.DataFrame(all_model_results)
    df = df.sort_values(by=['Approach', 'Shot', 'Reasoning', 'Model']).reset_index(drop=True)

    # Print the results grouped by Approach and Shot
    for (approach, shot), group_df in df.groupby(['Approach', 'Shot']):
        print(f"\n### Results for: {approach} ({shot})\n")
        print(group_df.to_markdown(index=False))
        print("\n")

if __name__ == "__main__":
    main()
