import os
import io
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Ensure stdout is UTF-8 (Windows-safe)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")  # base model
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/stoic-gpt2")  # fine-tuned LoRA

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# -------------------------------
# Load tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# -------------------------------
# Load base + LoRA model
# -------------------------------
print("Loading base model:", MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Applying LoRA adapters from:", MODEL_PATH)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

# Merge LoRA into the base model so pipeline works cleanly
print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

# -------------------------------
# Build pipeline
# -------------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# -------------------------------
# Run test prompt
# -------------------------------
if __name__ == "__main__":
    prompt = "How can I remain calm when I face setbacks?"
    outputs = generator(
        prompt,
        max_length=128,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.9
    )
    print("\nPrompt:", prompt)
    print("Reflection:", outputs[0]["generated_text"])

