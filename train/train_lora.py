"""
LoRA fine-tuning script (Hugging Face Trainer + PEFT)

Usage (example):
accelerate launch train/train_lora.py --data_path data/processed/train.jsonl --output_dir outputs/stoic-lora
"""
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='mistralai/Mixtral-8x7B-Instruct-v0.1')
    p.add_argument('--data_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_train_epochs', type=int, default=3)
    p.add_argument('--per_device_train_batch_size', type=int, default=4)
    p.add_argument('--max_length', type=int, default=512)
    return p.parse_args()

args = parse_args()

# load dataset
raw = load_dataset('json', data_files={'train': args.data_path})

def build_text(example):
    user_prompt = example['prompt']
    answer = example['response']

    # Mixtral / Mistral Instruct format
    return {
        "text": f"[INST] {user_prompt} [/INST] {answer}"
    }

raw = raw['train'].map(build_text)

# tokenizer & model
print('Loading tokenizer & model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map='auto')
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj','v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)

# tokenization
def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, max_length=args.max_length, padding='max_length')

tok = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names)

# set labels
tok = tok.map(lambda x: {'labels': x['input_ids']}, batched=True)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=args.num_train_epochs,
    fp16=True,
    save_total_limit=3,
    logging_steps=50,
    save_strategy='steps',
    save_steps=1000,
    evaluation_strategy='no',
    learning_rate=2e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok,
    data_collator=collator,
)

trainer.train()

# save
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print('Saved model to', args.output_dir)
