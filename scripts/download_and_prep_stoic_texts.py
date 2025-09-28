import os
import requests
import json
import random

# Gutenberg plain-text URLs (public domain)
TEXTS = {
    "aurelius_meditations.txt": "https://www.gutenberg.org/files/2680/2680-0.txt",
    "epictetus_enchiridion.txt": "https://www.gutenberg.org/files/45109/45109-0.txt",
    "epictetus_discourses.txt": "https://www.gutenberg.org/files/54632/54632-0.txt",
    "seneca_letters.txt": "https://www.gutenberg.org/files/56565/56565-0.txt"
}

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def clean_gutenberg_text(raw_text: str) -> str:
    """Remove Gutenberg header/footer boilerplate."""
    start_marker = "*** START OF THIS PROJECT GUTENBERG"
    end_marker = "*** END OF THIS PROJECT GUTENBERG"

    start_idx = raw_text.find(start_marker)
    if start_idx != -1:
        raw_text = raw_text[start_idx + len(start_marker):]

    end_idx = raw_text.find(end_marker)
    if end_idx != -1:
        raw_text = raw_text[:end_idx]

    return raw_text.strip()


def download_and_clean(name: str, url: str):
    print(f"Downloading {name} from {url} ...")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {url}")
        return None

    clean_text = clean_gutenberg_text(r.text)

    out_path = os.path.join(RAW_DIR, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"Saved {name} - {out_path}")
    return out_path


def build_pairs_from_text(text: str):
    """Create simple Q/A pairs in Mixtral format."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    pairs = []
    for p in paras:
        user_prompt = f"Provide a Stoic reflection on this passage:\n\n{p}"
        answer = f"{p}\n\nReflection: Focus only on what is within your control."
        formatted = f"[INST] {user_prompt} [/INST] {answer}"

        pairs.append({
            "prompt": user_prompt,
            "response": answer,
            "text": formatted
        })
    return pairs


def prepare_dataset():
    all_pairs = []
    for fname in os.listdir(RAW_DIR):
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            txt = f.read()
        pairs = build_pairs_from_text(txt)
        all_pairs.extend(pairs)

    print(f"Built {len(all_pairs)} Q/A pairs.")

    # Shuffle and split
    random.shuffle(all_pairs)
    n = len(all_pairs)
    train, valid, test = (
        all_pairs[: int(0.8 * n)],
        all_pairs[int(0.8 * n): int(0.9 * n)],
        all_pairs[int(0.9 * n):]
    )

    # Save JSONL
    for split, data in [("train", train), ("valid", valid), ("test", test)]:
        out_path = os.path.join(PROC_DIR, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} examples - {out_path}")


def main():
    for fname, url in TEXTS.items():
        download_and_clean(fname, url)

    prepare_dataset()
    print("All done! Dataset ready in data/processed/")


if __name__ == "__main__":
    main()
