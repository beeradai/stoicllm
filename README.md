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
# Windows
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

# Linux
gunicorn -k uvicorn.workers.UvicornWorker app:app -w 2 -b 0.0.0.0:8000
```
Sample request to above server:
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the Stoic view of fear?", "max_length":100}'
```
Swagger docs:
```bash
http://127.0.0.1:8000/docs
```
6. Docker build:
Local cpu-based
```bash
docker build \
  --build-arg BASE_IMAGE=python:3.10-slim \
  --build-arg REQ_FILE=requirements.txt \
  -t stoicllm-cpu .
```
Local gpu-based
```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --build-arg REQ_FILE=requirements-gpu.txt \
  -t stoicllm-gpu .
```
7. Docker deploy:
Local cpu-based
```bash
docker run -it --rm -p 8000:8000 stoicllm-cpu
```
Local gpu-based or on a standalone ec2 (requires additional config to ec2 before running):
```bash
docker run -it --rm --gpus all -p 8000:8000 stoicllm-gpu
```

Notes:
- This repo demonstrates a LoRA workflow; it uses 4-bit loading to reduce GPU memory requirements.
- Use only public-domain texts or content you have rights to.
- The included scripts are intentionally simple to be easy to modify for production usage.
