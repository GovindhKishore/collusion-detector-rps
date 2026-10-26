from pathlib import Path
import json
from collections import defaultdict, Counter
from setup.api_startup import Agent, BASE_PROMPT, parse_move

results_path = Path(__file__).parent / "excluded_char_collusion_results.json"

with open(results_path, "r") as fh:
    results = json.load(fh)

rounds = results["gemma,gemma,gemma,False"]
print("gemma,gemma,gemma,False \n")

for pos in range(3):
    print(pos, sum(r[pos] is None for r in rounds))
print("rounds with any None:", sum(None in r for r in rounds))

print("\n\n")

rounds = results["gemma,gemma,mistral,False"]
print("gemma,gemma,mistral,False \n")

for pos in range(3):
    print(pos, sum(r[pos] is None for r in rounds))
print("rounds with any None:", sum(None in r for r in rounds))

print("\n\n")

rounds = results["gemma,mistral,qwen,False"]
print("gemma,mistral,qwen,False \n")

for pos in range(3):
    print(pos, sum(r[pos] is None for r in rounds))
print("rounds with any None:", sum(None in r for r in rounds))


for key, value in results.items():
    print(key, len(value))