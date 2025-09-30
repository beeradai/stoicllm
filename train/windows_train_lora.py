import os
import sys
import warnings
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# -------------------------------
# Suppress noisy warnings
# -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# -------------------------------
# Windows-safe mode
# -------------------------------
if os.name == "nt":
    print("Running in Windows-safe mode (single process, no torch.distributed.elastic).")
    os.environ["ACCELERATE_DISABLE_RICH"] = "1"
    os.environ["ACCELERATE_USE_CPU"] = "false"  # keep GPU if available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")  # local test
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/stoic-gpt2")
DATA_PATH = os.environ.get("DATA_PATH", "data/processed/train.jsonl")

# -------------------------------
# Load dataset
# -------------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# -------------------------------
# Tokenizer + Model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# -------------------------------
# LoRA config
# -------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # GPT-2 safe
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -------------------------------
# Tokenization
# -------------------------------
def tokenize_fn(batch):
    texts = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "completion"])

# -------------------------------
# Trainer setup
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="no",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
# Train
# -------------------------------
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")

