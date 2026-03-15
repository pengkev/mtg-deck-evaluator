"""
Sorting, cleaning, organizing data across all the jsonls

Each dataset is compiled into one large, deduplicated, and shuffled file 
for Word2Vec embedding training.

Datasets:
ds1 = "../data/edh-decks.jsonl" (44k Moxfield, 14k MTGTop8)
ds2 = "../data/general-decks.jsonl" (MTGTop8: 26k cEDH, 71k Legacy, 97k Modern, 18k Vintage, 4k Pauper)
ds3 = "../data/large-moxfield-cEDH/official_harvest.jsonl" (130k Moxfield EDH, Brackets 1-5)
ds4 = "../data/large-mtgtop8-cEDH/mtgtop8_decks.jsonl" (24k recent MTGTop8 cEDH)

--- Schemas ---

ds1 & ds2 (Key-Value Dictionaries):
{
  "source": "mtgtop8-cEDH", 
  "deck_id": "deck_808872", 
  "mainboard": {"Birds of Paradise": 1, "Bloom Tender": 1, "...": "..."}, 
  "sideboard": {"Sisay, Weatherlight Captain": 1, "...": "..."}
}

ds3 (Lists of Dictionaries with 'n'):
{
  "id": "KkqaF1FAok6x...", 
  "name": "Vivi 2.0", 
  "user_bracket": null, 
  "auto_bracket": 4, 
  "mainboard": [{"n": "Gut Shot", "q": 1}, {"n": "Urza's Bauble", "q": 1}, "..."], 
  "commanders": [{"n": "Vivi Ornitier", "q": 1}]
}

ds4 (Lists of Dictionaries with 'name'):
{
  "deck_id": "818031", 
  "placement": 5, 
  "main": [{"name": "Altar of the Wretched", "qty": 1}, {"name": "Birgi", "qty": 1}, "..."], 
  "cmds": [{"name": "Dargo, the Shipwrecker", "qty": 1}, "..."]
}
"""

import json
import re
import random
from pathlib import Path

# Paths to the datasets
ds1_path = "../data/edh-decks.jsonl"
ds2_path = "../data/general-decks.jsonl"
ds3_path = "../data/large-moxfield-cEDH/official_harvest.jsonl"
ds4_path = "../data/large-mtgtop8-cEDH/mtgtop8_decks.jsonl"
out_path = "../data/embedding-corpus/unsupervised_megacorpus.jsonl"

RAW_BASIC_LANDS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest"
}

def normalize_card_name(name: str) -> str:
    """Lowercases the card and isolates the front face to unify formatting."""
    name = name.lower()
    front_face = re.split(r'\s*//?\s*', name)[0]
    return front_face.strip()

BASIC_LANDS = {normalize_card_name(c) for c in RAW_BASIC_LANDS}

def compile_megacorpus():
    seen_decks = set()
    all_unique_decks = []
    
    def process_and_add_deck(card_names):
        # Normalize and remove duplicate cards within the same deck
        clean_cards = {normalize_card_name(c) for c in card_names}
        
        # Filter out basic lands
        clean_cards = {c for c in clean_cards if c not in BASIC_LANDS}
        
        # Word2Vec needs context; skip tiny fragments
        if len(clean_cards) < 10:
            return
            
        # Sort to ensure identical decks always hash to the exact same tuple
        deck_tuple = tuple(sorted(clean_cards))
        
        # Global deduplication across all files
        if deck_tuple not in seen_decks:
            seen_decks.add(deck_tuple)
            all_unique_decks.append(list(deck_tuple))

    print("Processing ds1 and ds2 (Schema 1: Dicts)...")
    for path in [ds1_path, ds2_path]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    # Extract keys from mainboard and sideboard
                    cards = list(data.get("mainboard", {}).keys()) + list(data.get("sideboard", {}).keys())
                    process_and_add_deck(cards)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping.")

    print("Processing ds3 (Schema 2: List of Dicts with 'n')...")
    try:
        with open(ds3_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                cards = [item["n"] for item in data.get("mainboard", [])]
                cards += [item["n"] for item in data.get("commanders", [])]
                process_and_add_deck(cards)
    except FileNotFoundError:
        print(f"Warning: {ds3_path} not found. Skipping.")

    print("Processing ds4 (Schema 3: List of Dicts with 'name')...")
    try:
        with open(ds4_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                cards = [item["name"] for item in data.get("main", [])]
                cards += [item["name"] for item in data.get("cmds", [])]
                process_and_add_deck(cards)
    except FileNotFoundError:
        print(f"Warning: {ds4_path} not found. Skipping.")

    print(f"Total unique decks extracted: {len(all_unique_decks)}")
    
    # Critical step: Shuffle to prevent Gensim from suffering catastrophic forgetting
    print("Shuffling the megacorpus...")
    random.seed(42)
    random.shuffle(all_unique_decks)
    
    print(f"Writing to {out_path}...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for deck in all_unique_decks:
            # We save it under the "cards" key to match your updated MTGDeckCorpus iterator
            f.write(json.dumps({"cards": deck}) + "\n")
            
    print("Megacorpus compilation complete! Ready for Word2Vec.")

if __name__ == "__main__":
    compile_megacorpus()