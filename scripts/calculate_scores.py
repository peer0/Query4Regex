
import json
import re
import os
import csv
import glob
import sys

# Add the project root to the Python path to allow importing query4regex
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query4regex.eval.custom_equivalence import are_equivalent

def extract_generated_regex(generated_answer: str) -> str:
    """
    Extracts the regex from the oxed{...} format.
    """
    match = re.search(r'oxed\{(.*)\}', generated_answer, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def main():
    """
    Processes all generated result files, saves detailed evaluations,
    and calculates summary scores into a CSV file.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    generation_dir = os.path.join(base_dir, 'result', 'generation')
    evaluation_dir = os.path.join(base_dir, 'result', 'evaluation')
    csv_output_path = os.path.join(base_dir, 'result.csv')

    # Find all result files to process
    search_patterns = [
        os.path.join(generation_dir, 'zero-shot', '*.jsonl'),
        os.path.join(generation_dir, 'five-shot', '*.jsonl')
    ]
    
    all_files = []
    for pattern in search_patterns:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print("No result files found to process in result/generation/")
        return

    summary_scores = []

    for filepath in all_files:
        print(f"Processing {filepath}...")
        
        # Extract model name and methodology from the path
        methodology = os.path.basename(os.path.dirname(filepath))
        filename = os.path.basename(filepath)
        model_name = filename.replace('_False.jsonl', '')

        # Prepare output directory for detailed results
        detailed_output_dir = os.path.join(evaluation_dir, methodology)
        os.makedirs(detailed_output_dir, exist_ok=True)
        detailed_output_path = os.path.join(detailed_output_dir, filename)

        total_count = 0
        exact_match_count = 0
        equivalent_count = 0
        successfully_parsed_count = 0
        filtered_exact_match_count = 0
        filtered_equivalent_count = 0

        with open(filepath, 'r') as infile, open(detailed_output_path, 'w') as outfile:
            for line in infile:
                try:
                    data = json.loads(line)
                    total_count += 1

                    gold_regex = data.get('gold_regex', '')
                    generated_answer = data.get('generated_answer', '')
                    generated_regex = extract_generated_regex(generated_answer)

                    is_exact_match = (gold_regex == generated_regex)
                    
                    equivalence_result = are_equivalent(gold_regex, generated_regex)

                    is_parsable = equivalence_result is not None
                    is_equivalent = is_exact_match or equivalence_result is True

                    if is_exact_match:
                        exact_match_count += 1
                    
                    if is_equivalent:
                        equivalent_count += 1

                    if is_parsable:
                        successfully_parsed_count += 1
                        if is_exact_match:
                            filtered_exact_match_count += 1
                        if is_equivalent:
                            filtered_equivalent_count += 1

                    # Write detailed result to the new jsonl file
                    detailed_result = {
                        'gold_regex': gold_regex,
                        'generated_regex': generated_regex,
                        'exact_match': is_exact_match,
                        'equivalent': is_equivalent,
                        'parsable': is_parsable
                    }
                    outfile.write(json.dumps(detailed_result) + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {filepath}: {line.strip()}")
                    continue
        
        # Calculate scores
        exact_match_score = (exact_match_count / total_count) if total_count > 0 else 0
        equivalence_score = (equivalent_count / total_count) if total_count > 0 else 0
        successfully_parsed_ratio = (successfully_parsed_count / total_count) if total_count > 0 else 0
        filtered_exact_match_score = (filtered_exact_match_count / successfully_parsed_count) if successfully_parsed_count > 0 else 0
        filtered_equivalence_score = (filtered_equivalent_count / successfully_parsed_count) if successfully_parsed_count > 0 else 0

        summary_scores.append({
            'model_name': model_name,
            'methodology': methodology,
            'exact_match_score': f"{exact_match_score:.4f}",
            'equivalence_score': f"{equivalence_score:.4f}",
            'successfully_parsed_ratio': f"{successfully_parsed_ratio:.4f}",
            'filtered_exact_match_score': f"{filtered_exact_match_score:.4f}",
            'filtered_equivalence_score': f"{filtered_equivalence_score:.4f}"
        })

    # Write summary scores to CSV
    if summary_scores:
        print(f"Writing summary scores to {csv_output_path}...")
        with open(csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['model_name', 'methodology', 'exact_match_score', 'equivalence_score', 'successfully_parsed_ratio', 'filtered_exact_match_score', 'filtered_equivalence_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_scores)
        print("Done.")
    else:
        print("No scores to write.")

if __name__ == "__main__":
    main()
