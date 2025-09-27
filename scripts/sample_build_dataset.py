# Creates a small sample dataset in examples/sample_train.jsonl for quick testing
import json, os

os.makedirs('examples', exist_ok=True)
sample = []
for i in range(500):
    prompt = f"User: I'm anxious about giving a talk (sample {i}). How should I think about it?"
    resp = "StoicLLM: Remember that the outcome is not fully yours to command. Prepare well, speak with honesty, and accept the rest. (sample)"
    sample.append({"id": f"s{i}", "prompt": prompt, "response": resp})

with open('examples/sample_train.jsonl','w',encoding='utf-8') as fh:
    for obj in sample:
        fh.write(json.dumps(obj,ensure_ascii=False) + '\n')
print('wrote examples/sample_train.jsonl')