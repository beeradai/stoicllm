import argparse
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# -----------------------------
# Helpers
# -----------------------------
def auto_cast(val):
    if isinstance(val, str):
        try:
            if "." in val or "e" in val.lower():
                return float(val)
            return int(val)
        except Exception:
            return val
    return val

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return {k: auto_cast(v) for k, v in config.items()}

# -----------------------------
# LoRA target module detection
# -----------------------------
MODEL_TARGET_MODULES = {
    "gpt2": ["c_attn", "c_proj"],
    "meta-llama": ["q_proj", "v_proj"],
    "mistralai": ["q_proj", "v_proj"],
}

def get_target_modules(model_name):
    for key, modules in MODEL_TARGET_MODULES.items():
        if key in model_name.lower():
            return modules
    return ["c_attn", "c_proj"]  # fallback

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path.")
    parser.add_argument("--data_path", type=str, default="data/processed/train.jsonl", help="Training data path.")
    parser.add_argument("--output_dir", type=str, default="outputs/stoic-lora", help="Output directory.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()

# -----------------------------
# Main training logic
# -----------------------------
def main():
    args = parse_args()

    # Load config if provided
    config = {}
    if args.config_file:
        config = load_config(args.config_file)

    # Merge config + CLI args (CLI overrides YAML)
    merged_args = vars(args)
    merged_args.update(config)

    model_name = merged_args["model_name"]
    data_path = merged_args["data_path"]
    output_dir = merged_args["output_dir"]

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=get_target_modules(model_name),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Load and split dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 90% train / 10% validation

    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_train = dataset["train"].map(encode, batched=True)
    tokenized_eval = dataset["test"].map(encode, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(merged_args["per_device_train_batch_size"]),
        num_train_epochs=int(merged_args["num_train_epochs"]),
        learning_rate=float(merged_args["learning_rate"]),
        evaluation_strategy="epoch",       # evaluate after each epoch
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
