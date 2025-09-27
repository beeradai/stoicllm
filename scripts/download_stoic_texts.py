import os
import requests

# Gutenberg plain-text URLs (public domain)
TEXTS = {
    "aurelius_meditations.txt": "https://www.gutenberg.org/files/2680/2680-0.txt",
    "epictetus_enchiridion.txt": "https://www.gutenberg.org/files/45109/45109-0.txt",
    "epictetus_discourses.txt": "https://www.gutenberg.org/files/54632/54632-0.txt",
    "seneca_letters.txt": "https://www.gutenberg.org/files/56565/56565-0.txt"
}

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def clean_gutenberg_text(raw_text: str) -> str:
    """
    Remove Gutenberg header/footer boilerplate.
    """
    start_marker = "*** START OF THIS PROJECT GUTENBERG"
    end_marker = "*** END OF THIS PROJECT GUTENBERG"

    # Find content boundaries
    start_idx = raw_text.find(start_marker)
    if start_idx != -1:
        raw_text = raw_text[start_idx + len(start_marker):]

    end_idx = raw_text.find(end_marker)
    if end_idx != -1:
        raw_text = raw_text[:end_idx]

    # Basic cleanup
    raw_text = raw_text.strip()
    return raw_text


def download_and_clean(name: str, url: str):
    print(f"Downloading {name} from {url} ...")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {url}")
        return

    clean_text = clean_gutenberg_text(r.text)

    out_path = os.path.join(RAW_DIR, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"Saved {name} - {out_path}")


def main():
    for fname, url in TEXTS.items():
        download_and_clean(fname, url)


if __name__ == "__main__":
    main()
