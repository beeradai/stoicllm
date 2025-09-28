# StoicLLM - Training Repo

This repo contains scripts and sample data to fine-tune a base causal LM into StoicLLM using LoRA (PEFT) and Hugging Face tools.
It is intended as a practical, runnable starting point. Edit hyperparameters and data handling to match your needs.

Quick start (local or cloud GPU):

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Put source texts into `data/raw/` (public-domain Stoic texts, essays, etc.). You can use the provided `examples/sample_train.jsonl` for quick tests.

3. Generate processed JSONL dataset:

```bash
python scripts/data_prep.py --input_dir data/raw --output data/processed/train.jsonl
```

4. Run training (example using `accelerate`):

```bash
accelerate launch --config_file train/accelerate_config.yaml train/train_lora.py --data_path data/processed/train.jsonl --output_dir outputs/stoic-lora
```

5. Evaluate:

```bash
python eval/eval_model.py --model outputs/stoic-lora --data data/processed/valid.jsonl
```

6. Quick deploy (local):

```bash
python deploy/app.py --model_dir outputs/stoic-lora --port 8000
```

Notes:
- This repo demonstrates a LoRA workflow; it uses 8-bit loading to reduce GPU memory requirements.
- Use only public-domain texts or content you have rights to.
- The included scripts are intentionally simple to be easy to modify for production usage.
