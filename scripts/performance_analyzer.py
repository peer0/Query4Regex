
import json
import re
import os
import glob
import sys
from collections import defaultdict

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query4regex.eval.custom_equivalence import are_equivalent, is_valid_standard_regex

# Custom operators for complexity analysis
CUSTOM_OPERATORS = ['&', '~', '{'] # We only check for the opening brace

def extract_generated_regex(generated_answer: str) -> str:
    """
    Extracts the regex from the oxed{...} format.
    """
    match = re.search(r'oxed(.*)', generated_answer, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def count_operators(regex_str: str) -> int:
    """
    Counts the number of custom operators in a regex string.
    """
    return sum(regex_str.count(op) for op in CUSTOM_OPERATORS)

def get_complexity_group(operator_count: int) -> str:
    """
    Assigns a complexity group based on the number of operators.
    """
    if operator_count <= 1:
        return "0-1 (Low)"
    elif 2 <= operator_count <= 3:
        return "2-3 (Medium)"
    else:
        return "4+ (High)"

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Paths to the data files
    search_patterns = [
        os.path.join(base_dir, 'result', 'generation', 'nl2regex', '**', '*.jsonl'),
        os.path.join(base_dir, 'result', 'generation', 'nl2dsl2regex', '**', '*.jsonl')
    ]

    all_files = []
    for pattern in search_patterns:
        all_files.extend(glob.glob(pattern, recursive=True))

    if not all_files:
        print("No result files found.")
        return

    # --- Data Collection ---
    all_results = []
    for filepath in all_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_cases = len(all_results)
    
    # --- Analysis ---
    parser_success_count = 0
    regex_validity_count = 0
    total_score = 0
    
    complexity_data = defaultdict(lambda: {'count': 0, 'total_score': 0})

    for data in all_results:
        gold_regex = data.get('gold_regex', '')
        generated_answer = data.get('generated_answer', '')
        generated_regex = extract_generated_regex(generated_answer)

        # 1. Internal Parser Success
        # We use our custom equivalence check which returns None on parsing failure.
        equivalence_result = are_equivalent(gold_regex, generated_regex)
        is_parsable = equivalence_result is not None

        if is_parsable:
            parser_success_count += 1

            # 2. Standard Regex Validity (Post-Parsing)
            if is_valid_standard_regex(generated_regex):
                regex_validity_count += 1

        # 3. Score Calculation (Equivalence Score)
        score = 1 if equivalence_result is True else 0
        total_score += score

        # 4. Complexity Analysis
        operator_count = count_operators(gold_regex)
        group = get_complexity_group(operator_count)
        complexity_data[group]['count'] += 1
        complexity_data[group]['total_score'] += score

    # --- Calculate Final Metrics ---
    parsing_success_rate = (parser_success_count / total_cases) * 100 if total_cases > 0 else 0
    # Of those that parsed, how many are valid
    regex_validity_rate = (regex_validity_count / parser_success_count) * 100 if parser_success_count > 0 else 0
    overall_average_score = (total_score / total_cases) if total_cases > 0 else 0

    # --- Generate Report ---
    print("### Query4Regex Performance Measurement Report")
    print("\n#### 1. Summary")
    print(f"  * **Overall Parsing Success Rate**: {parsing_success_rate:.1f}%" )
    print(f"  * **Regex Validity Rate (Post-Parsing)**: {regex_validity_rate:.1f}%" )
    print(f"  * **Overall Average Score**: {overall_average_score:.3f}")

    print("\n#### 2. Detailed Analysis")
    print("\n**A. Parsing and Validity Results**")
    print("| Metric | Success | Failure | Rate |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| **Internal Parser Success** | {parser_success_count} cases | {total_cases - parser_success_count} cases | {parsing_success_rate:.1f}% |")
    print(f"| **Standard Regex Validity** | {regex_validity_count} cases | {parser_success_count - regex_validity_count} cases | {regex_validity_rate:.1f}% |")

    print("\n**B. Performance Analysis by Operator Complexity**")
    print("| Number of Operators (Complexity) | # of Test Cases | Average Score |")
    print("| :--- | :--- | :--- |")

    # Sort complexity groups for consistent reporting
    sorted_groups = ["0-1 (Low)", "2-3 (Medium)", "4+ (High)"]
    for group_name in sorted_groups:
        data = complexity_data[group_name]
        count = data['count']
        avg_score = (data['total_score'] / count) if count > 0 else 0
        print(f"| **{group_name}** | {count} | {avg_score:.3f} |")

if __name__ == "__main__":
    main()
