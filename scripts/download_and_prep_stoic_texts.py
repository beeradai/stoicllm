import os
import re
import json
import random
import requests

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.jsonl")
VALID_FILE = os.path.join(PROCESSED_DIR, "valid.jsonl")

# Example public domain sources
STOIC_TEXTS = {
    "meditations.txt": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",  # Marcus Aurelius - Meditations
    "enchiridion.txt": "https://www.gutenberg.org/cache/epub/45109/pg45109.txt",  # Epictetus - Enchiridion
    "seneca_letters.txt": "https://www.gutenberg.org/cache/epub/56376/pg56376.txt",  # Seneca - Letters
    "epictetus_discourses.txt": "https://www.gutenberg.org/files/54632/54632-0.txt", # Epictetus - Discourses
}

def download_texts():
    """Download Stoic texts into data/raw if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)

    for filename, url in STOIC_TEXTS.items():
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} from {url} ...")
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            print(f"Found existing {filename}, skipping download.")


def clean_line(line: str) -> str:
    """Remove boilerplate, headers, and unwanted patterns."""
    line = line.strip()

    # Skip empty or very short lines
    if len(line) < 40:
        return ""

    # Remove Project Gutenberg mentions
    if "gutenberg" in line.lower():
        return ""

    # Remove chapter/letter/book headings
    if re.match(r"^(chapter|book|letter)\b", line.lower()):
        return ""

    # Skip lines with only numbers or Roman numerals
    if re.match(r"^[ivxlcdm0-9]+$", line.lower()):
        return ""

    # Keep moderately sized sentences (avoid huge blocks)
    if len(line) > 500:
        return ""

    return line


def build_dataset(input_files, train_file, val_file, split_ratio=0.9):
    """Convert raw Stoic texts into a cleaned JSONL dataset with train/val split."""
    os.makedirs(os.path.dirname(train_file), exist_ok=True)

    records = []
    for f in input_files:
        with open(f, "r", encoding="utf-8") as inp:
            text = inp.read().splitlines()
            for line in text:
                cleaned = clean_line(line)
                if cleaned:
                    records.append({"prompt": "Stoic Reflection:", "completion": cleaned})

    random.shuffle(records)
    split_idx = int(len(records) * split_ratio)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dataset built: {len(train_records)} train / {len(val_records)} val examples")


if __name__ == "__main__":
    print("Preparing Stoic texts...")
    download_texts()

    input_files = [os.path.join(RAW_DIR, f) for f in STOIC_TEXTS.keys()]
    build_dataset(input_files, TRAIN_FILE, VALID_FILE)

