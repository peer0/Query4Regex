
import json
from collections import Counter

def get_operation_statistics(file_path):
    op_counts = Counter()
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            ops_dsl = data.get("ops_dsl", "")
            num_ops = len(ops_dsl.split(";"))
            op_counts[num_ops] += 1
    
    print("Operation Number Statistics:")
    for num_ops, count in sorted(op_counts.items()):
        print(f"  Number of operations {num_ops}: {count} data instances")

# Get and print the statistics
get_operation_statistics("/home/greg/Query4Regex/data/test.jsonl")
