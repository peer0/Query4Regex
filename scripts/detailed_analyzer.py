import json
import re
import os
import glob
import sys
from collections import defaultdict
import pandas as pd
import csv

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query4regex.eval.custom_equivalence import are_equivalent

def extract_generated_regex(generated_answer: str) -> str:
    match = re.search(r'oxed\{(.*)\}', generated_answer, re.DOTALL)
    return match.group(1) if match else ""

def count_ops_from_instruction(instruction: str) -> int:
    if not instruction:
        return 0
    # Count sentences as a proxy for operations
    return len(re.split(r'[.!?]\s*', instruction.strip())) -1

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
    op_counts = defaultdict(lambda: {'count': 0, 'correct': 0})
    filtered_op_counts = defaultdict(lambda: {'count': 0, 'correct': 0})
    detailed_results = []

    for data in results:
        gold_regex = data.get('gold_regex', '')
        generated_answer = data.get('generated_answer', '')
        generated_regex = extract_generated_regex(generated_answer)
        instruction = data.get('instruction', '')
        
        equivalence_result = are_equivalent(gold_regex, generated_regex)
        is_parsable = equivalence_result is not None
        score = 1 if equivalence_result is True else 0

        num_ops = count_ops_from_instruction(instruction)
        op_counts[num_ops]['count'] += 1
        op_counts[num_ops]['correct'] += score

        if is_parsable:
            parsing_success_count += 1
            filtered_op_counts[num_ops]['count'] += 1
            filtered_op_counts[num_ops]['correct'] += score

        detailed_results.append({
            'gold_regex': gold_regex,
            'generated_regex': generated_regex,
            'exact_match': gold_regex == generated_regex,
            'equivalent': equivalence_result is True,
            'parsable': is_parsable
        })

    # Calculate scores
    total_score = sum(v['correct'] for v in op_counts.values())
    parsing_rate = (parsing_success_count / total_cases) * 100 if total_cases > 0 else 0
    avg_score = (total_score / total_cases) * 100 if total_cases > 0 else 0
    filtered_avg_score = (sum(v['correct'] for v in filtered_op_counts.values()) / parsing_success_count) * 100 if parsing_success_count > 0 else 0

    op_scores = {}
    filtered_op_scores = {}
    op_keys = sorted(op_counts.keys())

    for op_num in op_keys:
        count = op_counts[op_num]['count']
        correct = op_counts[op_num]['correct']
        op_scores[op_num] = (correct / count) * 100 if count > 0 else 0

        filtered_count = filtered_op_counts[op_num]['count']
        filtered_correct = filtered_op_counts[op_num]['correct']
        filtered_op_scores[op_num] = (filtered_correct / filtered_count) * 100 if filtered_count > 0 else 0

    analysis_summary = {
        'parsing_rate': parsing_rate,
        'average_score': avg_score,
        'filtered_average_score': filtered_avg_score,
    }
    for op_num in op_keys:
        analysis_summary[f'op_{op_num}_score'] = op_scores.get(op_num, 0)
        analysis_summary[f'filtered_op_{op_num}_score'] = filtered_op_scores.get(op_num, 0)

    return analysis_summary, detailed_results

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generation_dir = os.path.join(base_dir, 'result', 'generation')
    evaluation_dir = os.path.join(base_dir, 'result', 'evaluation')
    csv_output_path = os.path.join(base_dir, 'result.csv')
    
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
            approach = parts[-3]
            shot = parts[-2]
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

        result_row = {
            'Approach': approach,
            'Shot': shot,
            'Model': model_name,
            'Reasoning': is_reasoning,
            'Parsing Rate (%)': f"{analysis['parsing_rate']:.2f}",
            'Avg Score (%)': f"{analysis['average_score']:.2f}",
            'Filtered Avg Score (%)': f"{analysis['filtered_average_score']:.2f}",
        }
        for i in [1, 2, 3]:
            result_row[f'Ops={i} Score (%)'] = f"{analysis.get(f'op_{i}_score', 0):.2f}"
            result_row[f'Filtered Ops={i} Score (%)'] = f"{analysis.get(f'filtered_op_{i}_score', 0):.2f}"

        all_model_results.append(result_row)

    # Write to CSV
    if all_model_results:
        headers = ['Approach', 'Shot', 'Model', 'Reasoning', 'Parsing Rate (%)', 'Avg Score (%)', 
                   'Ops=1 Score (%)', 'Ops=2 Score (%)', 'Ops=3 Score (%)', 
                   'Filtered Avg Score (%)', 'Filtered Ops=1 Score (%)', 'Filtered Ops=2 Score (%)', 'Filtered Ops=3 Score (%)']
        
        df = pd.DataFrame(all_model_results)
        df.to_csv(csv_output_path, index=False, columns=headers)

        print(f"Results saved to {csv_output_path}")

if __name__ == "__main__":
    main()
