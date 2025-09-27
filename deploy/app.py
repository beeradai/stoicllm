"""
Minimal FastAPI server for demo inference. NOT production hardened.

Run:
python deploy/app.py --model_dir outputs/stoic-lora --port 8000
"""
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Query(BaseModel):
    prompt: str

def start(model_dir, host='0.0.0.0', port=8000):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto')
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    app = FastAPI()

    @app.post('/chat')
    async def chat(q: Query):
        if len(q.prompt) > 2000:
            raise HTTPException(status_code=400, detail='Prompt too long')
        out = gen(q.prompt, max_new_tokens=256, do_sample=False)
        return { 'reply': out[0]['generated_text'] }

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    start(args.model_dir, port=args.port)
