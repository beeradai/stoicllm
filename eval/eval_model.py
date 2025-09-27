"""
Basic evaluation: loads model and calculates perplexity on a JSONL dataset, and runs sample generations.
"""
import argparse
import math
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def perplexity(model, tokenizer, texts, max_length=512):
    enc = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    input_ids = enc['input_ids'].to(model.device)
    attention_mask = enc['attention_mask'].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss.item()
    return math.exp(loss)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    args = p.parse_args()

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')

    ds = load_dataset('json', data_files={'test': args.data})['test']
    samples = [row['prompt'] + '\\n\\n' + row['response'] for i,row in enumerate(ds) if i<64]

    ppl = perplexity(model, tokenizer, samples)
    print('Estimated perplexity (64 samples):', ppl)

    # generate sample responses
    prompt = "User: I'm anxious about public speaking.\\n\\nRespond as StoicLLM:" 
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    out = model.generate(input_ids, max_new_tokens=150, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
