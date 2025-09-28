# StoicLLM - Training Repo

This repo contains scripts and sample data to fine-tune a base causal LM into StoicLLM using LoRA (PEFT) and Hugging Face tools.
It is intended as a practical, runnable starting point. Edit hyperparameters and data handling to match your needs.

Quick start (local or cloud GPU):

1. Install requirements:

```bash
# Local gpt-2
pip install -r requirements.txt

# Cloud mixtral
pip install -r requirements-gpu.txt
```

2. Put source texts into `data/raw/` (public-domain Stoic texts, essays, etc.). You can use the provided `examples/sample_train.jsonl` for quick tests.
```bash
python download_stoic_texts.py
```

3. Generate processed JSONL dataset:

```bash
python scripts/data_prep.py --input_dir data/raw --output data/processed/train.jsonl
```
Or combine steps 2 & 3 with:
```bash
python download_and_prep_stoic_texts.py
```

4. Run training & evaluation:

```bash
# Local gpt-2
python train/train_lora.py --config_file train/configs/local_gpt2.yaml

# Cloud mixtral
accelerate launch train/train_lora.py --config_file train/configs/cloud_mixtral.yaml
```

5. Quick deploy (local):

```bash
python deploy/app.py --model_dir outputs/stoic-lora --port 8000
```

Notes:
- This repo demonstrates a LoRA workflow; it uses 4-bit loading to reduce GPU memory requirements.
- Use only public-domain texts or content you have rights to.
- The included scripts are intentionally simple to be easy to modify for production usage.
