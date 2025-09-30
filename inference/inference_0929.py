import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# Select device automatically
# ============================================================
device = 0 if torch.cuda.is_available() else -1

# ============================================================
# Paths
# ============================================================
BASE_MODEL = os.getenv("BASE_MODEL", "gpt2")  # fallback for testing
MODEL_PATH = os.getenv("MODEL_PATH", "outputs/stoic-gpt2")

def load_model():
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    try:
        print(f"Applying LoRA adapters from: {MODEL_PATH}")
        # Load and merge LoRA adapters
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"⚠️ Could not load LoRA adapters from {MODEL_PATH}, using base model only. Error: {e}")
        model = base_model

    return model, tokenizer

def run_inference(prompt: str):
    model, tokenizer = load_model()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device=0 if torch.cuda.is_available() else -1
    )

    outputs = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2
    )

    return outputs[0]["generated_text"]

if __name__ == "__main__":
    test_prompt = "What is the essence of Stoicism?"
    result = run_inference(test_prompt)
    print("\n=== Generated Response ===\n")
    print(result)

