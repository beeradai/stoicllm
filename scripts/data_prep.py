"""
Simple data preparation script.

- Reads text files from input_dir (utf-8). Each file may contain multiple paragraphs.
- Produces instruction-response JSONL with `prompt` and `response` fields.
- Also outputs `train.jsonl`, `valid.jsonl` and `test.jsonl` split.

Customize for production data cleaning and annotation workflows.
"""
import os
import argparse
import json
import re
from glob import glob
from random import shuffle, seed

seed(42)

RE_PROMPT = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s.replace('\u2019', "'")
    s = s.replace('\u201c', '"').replace('\u201d','"')
    s = s.strip()
    s = RE_PROMPT.sub(' ', s)
    return s

def build_pairs_from_text(text: str, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
    # Naive heuristic: split into paragraphs and create Q/A style pairs
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
    pairs = []
    for p in paras:
        cp = clean_text(p)

        user_prompt = f"Provide a brief Stoic reflection on the following quote:\n\n{cp}"
        answer = f"{cp}\n\nReflection: Consider what is within your control. Ask: what is my duty here?"

        # --- Model-specific formatting ---
        if "mixtral" in model_name.lower() or "mistral" in model_name.lower():
            formatted = f"[INST] {user_prompt} [/INST] {answer}"
        elif "llama" in model_name.lower():
            formatted = f"<s>[INST] {user_prompt} [/INST] {answer}</s>"
        else:
            formatted = user_prompt + "\n\n" + answer

        pairs.append({"prompt": user_prompt, "response": answer, "text": formatted})
    return pairs


def main(input_dir, output_dir, min_examples=200, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    files = glob(os.path.join(input_dir, '*'))
    out = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                txt = fh.read()
            pairs = build_pairs_from_text(txt, model_name)
            out.extend(pairs)
        except Exception as e:
            print(f"skipping {f}: {e}")

    shuffle(out)
    # ensure minimum examples by duplicating with small variation if needed
    if len(out) > 0 and len(out) < min_examples:
        add = []
        for i in range(min_examples - len(out)):
            p = out[i % len(out)].copy()
            p['prompt'] = p['prompt'] + ' (variant)'
            add.append(p)
        out.extend(add)

    if len(out) == 0:
        raise SystemExit("No source data found in input_dir. Put plain text files into the folder, or use the sample dataset in examples/.")

    # split
    n = len(out)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = out[:n_train]
    val = out[n_train:n_train+n_val]
    test = out[n_train+n_val:]

    os.makedirs(output_dir, exist_ok=True)

    def write_list(l, path):
        with open(path, 'w', encoding='utf-8') as fh:
            for i, obj in enumerate(l):
                obj_out = {"id": f"ex_{i}", "prompt": obj['prompt'], "response": obj['response']}
                fh.write(json.dumps(obj_out, ensure_ascii=False) + "\n")

    write_list(train, os.path.join(output_dir, 'train.jsonl'))
    write_list(val, os.path.join(output_dir, 'valid.jsonl'))
    write_list(test, os.path.join(output_dir, 'test.jsonl'))
    print(f"Wrote {len(train)} train, {len(val)} valid, {len(test)} test to {output_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True, help='Directory with source text files')
    p.add_argument('--output_dir', required=True, help='Directory to write train/valid/test JSONL files')
    p.add_argument('--min_examples', type=int, default=200)
    args = p.parse_args()
    main(args.input_dir, args.output_dir, args.min_examples)
