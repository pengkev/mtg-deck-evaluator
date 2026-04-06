import json
import re
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset

# --- Configuration ---
EMBEDDING_MODEL = Path("../data/general-item2vec_mtg.model")
INPUT_EDH = Path("../data/edh-decks.jsonl")

MAX_LEN = 115

BASIC_LANDS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest"
}

BRACKET_RE = re.compile(r"bracket[-_ ]?([1-5])")

# --- Functions & Classes ---
def load_gensim_vocab(gensim_model_path: Path):
    """Loads just the vocabulary dictionary, skipping the heavy weight matrices."""
    print(f"Loading vocabulary from {gensim_model_path}...")
    g_model = Word2Vec.load(str(gensim_model_path))

    gensim_words = g_model.wv.index_to_key
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    for i, word in enumerate(gensim_words, start=2):
        vocab[word] = i
        
    print(f"Loaded {len(vocab)} unique words into vocabulary.")
    return vocab


def parse_bracket_label(source: str):
    match = BRACKET_RE.search(source.lower())
    if not match:
        return None
    return int(match.group(1)) - 1


class MTGDeckDataset(Dataset):
    def __init__(self, jsonl_filepath: Path, vocab, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.vocab = vocab
        self.inputs = []
        self.labels = []

        print(f"Parsing decks from {jsonl_filepath}...")
        with open(jsonl_filepath, "r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue

                data = json.loads(line)
                label = parse_bracket_label(data.get("source", ""))
                if label is None:
                    continue

                mainboard = list(data.get("mainboard", {}).keys())
                sideboard = list(data.get("sideboard", {}).keys())
                cards = [card for card in (mainboard + sideboard) if card not in BASIC_LANDS]
                
                if len(cards) < 10:
                    continue

                deck_ids = [self.vocab.get(card, 1) for card in cards[: self.max_len]]
                if len(deck_ids) < self.max_len:
                    deck_ids += [0] * (self.max_len - len(deck_ids))

                self.inputs.append(deck_ids)
                self.labels.append(label)

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Dataset ready: {len(self.inputs)} labeled decks.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# --- The Diagnostic ---
def check_blindness():
    vocab = load_gensim_vocab(EMBEDDING_MODEL)
    dataset = MTGDeckDataset(INPUT_EDH, vocab, max_len=MAX_LEN)
    
    # Evaluate the entire dataset tensor at once
    all_inputs = dataset.inputs
    
    # Ignore padding (ID 0)
    valid_mask = all_inputs != 0
    valid_cards = all_inputs[valid_mask]
    
    total_valid_cards = valid_cards.numel()
    total_unks = (valid_cards == 1).sum().item()
    
    unk_ratio = (total_unks / total_valid_cards) * 100
    
    print("\n" + "="*45)
    print(f"Total Card Slots Checked: {total_valid_cards:,}")
    print(f"Total <UNK> Tokens (ID 1): {total_unks:,}")
    print(f"UNK Ratio:                 {unk_ratio:.2f}%")
    print("="*45)

if __name__ == "__main__":
    check_blindness()