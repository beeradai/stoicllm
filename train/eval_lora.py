import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import math

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="YAML config file for evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config_file)

    model_name = config["model_name"]
    data_path = config["data_path"]

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load dataset (reuse same train/test split)
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_eval = dataset["test"].map(encode, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Dummy training args for evaluation
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    results = trainer.evaluate()
    perplexity = math.exp(results["eval_loss"])
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()
