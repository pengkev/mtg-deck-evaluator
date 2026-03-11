"""
Compresses all deck files in data/ into a single gzipped JSONL archive.

Output: data/decks.jsonl.gz
Each line is a JSON object:
  {
    "source": "moxfield-edh-bracket-1",
    "deck_id": "___zvKuXW0SiydqN1QCUwQ",
    "mainboard": {"Sol Ring": 1, "Command Tower": 1, ...},
    "sideboard": {"Obuun, Mul Daya Ancestor": 1}
  }

Run from the repo root:
  python helpers/compressor.py
"""

import gzip
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "decks.jsonl.gz")

CARD_LINE_RE = re.compile(r"^(\d+)\s+(.+)$")


def parse_deck(filepath: str) -> tuple[dict, dict]:
    """Parse a deck file and return (mainboard, sideboard) as {card_name: count} dicts."""
    mainboard: dict[str, int] = {}
    sideboard: dict[str, int] = {}
    in_sideboard = False

    with open(filepath, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower() == "sideboard":
                in_sideboard = True
                continue
            match = CARD_LINE_RE.match(line)
            if match:
                count = int(match.group(1))
                name = match.group(2).strip()
                target = sideboard if in_sideboard else mainboard
                target[name] = target.get(name, 0) + count

    return mainboard, sideboard


def compress_data():
    sources = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    total = 0
    skipped = 0

    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as out:
        for source in sorted(sources):
            source_dir = os.path.join(DATA_DIR, source)
            files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
            for filename in sorted(files):
                filepath = os.path.join(source_dir, filename)
                try:
                    mainboard, sideboard = parse_deck(filepath)
                except Exception as e:
                    print(f"  [skip] {source}/{filename}: {e}")
                    skipped += 1
                    continue

                if not mainboard:
                    skipped += 1
                    continue

                record = {
                    "source": source,
                    "deck_id": os.path.splitext(filename)[0],
                    "mainboard": mainboard,
                    "sideboard": sideboard,
                }
                out.write(json.dumps(record) + "\n")
                total += 1

    print(f"Done. Wrote {total} decks to {OUTPUT_PATH} ({skipped} skipped).")


if __name__ == "__main__":
    compress_data()
