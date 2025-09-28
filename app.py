import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Detect device
if torch.cuda.is_available():
    device = "cuda"
    MODEL_PATH = "outputs/stoic-mixtral"   # cloud model
elif torch.backends.mps.is_available():  # Apple Silicon
    device = "mps"
    MODEL_PATH = "outputs/stoic-gpt2"     # fallback to GPT-2
else:
    device = "cpu"
    MODEL_PATH = "outputs/stoic-gpt2"     # local model

print(f"⚡ Using device: {device}, loading model from: {MODEL_PATH}")

# Load model/tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto" if device != "cpu" else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)

# FastAPI app
app = FastAPI(title="StoicLLM API")

# Request body
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

# Endpoint
@app.post("/generate")
def generate_text(req: GenerateRequest):
    outputs = pipe(
        req.prompt,
        max_length=req.max_length,
        do_sample=True,
        top_p=req.top_p,
        temperature=req.temperature
    )
    return {"response": outputs[0]["generated_text"]}

