"""
Regression Dataset Compiler (Brackets 1-5)

Compiles a unified, deduplicated dataset for training the Ordinal Regression model.
Filters out all 60-card formats. Automatically assigns Bracket 5 to tournament cEDH lists.
Prioritizes user-assigned labels over auto-labels during deduplication.

Datasets:
ds1 = "../data/edh-decks.jsonl" (Extracts only cEDH/MTGTop8)
ds2 = "../data/general-decks.jsonl" (Extracts only cEDH/MTGTop8)
ds3 = "../data/large-moxfield-cEDH/official_harvest.jsonl" (130k Moxfield EDH)
ds4 = "../data/large-mtgtop8-cEDH/mtgtop8_decks.jsonl" (24k MTGTop8 cEDH)

--- Schemas ---
ds1 & ds2 (Dicts): {"source": "mtgtop8-cEDH", "deck_id": "...", "mainboard": {...}, "sideboard": {...}}
ds3 (Lists of Dicts): {"id": "...", "user_bracket": null, "auto_bracket": 4, "mainboard": [{"n": "Card", "q": 1}], "commanders": [...]}
ds4 (Lists of Dicts): {"deck_id": "...", "deck_url": "...", "main": [{"name": "Card", "qty": 1}], "cmds": [...]}
"""

import json
import re
import random
from pathlib import Path

# Paths to the datasets
ds1_path = "../data/edh_decks.jsonl"
ds2_path = "../data/general_decks.jsonl"
ds3_path = "../data/large-moxfield-cEDH/official_harvest.jsonl"
ds4_path = "../data/large-mtgtop8-cEDH/mtgtop8_decks.jsonl"
out_path = "../data/big-corpus/supervised_megacorpus.jsonl"

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

def compile_regression_dataset():
    # Maps deck_tuple -> deck_data_dict
    seen_decks = {}
    
    def process_and_add_deck(deck_id, url, card_names, bracket, is_autobracket):
        clean_cards = {normalize_card_name(c) for c in card_names}
        clean_cards = {c for c in clean_cards if c not in BASIC_LANDS}
        
        # Require at least 50 non-basic cards to be considered a valid EDH deck
        if len(clean_cards) < 50:
            return
            
        deck_tuple = tuple(sorted(clean_cards))
        
        # Quality score: Human/Tournament (2) overrides Auto-heuristic (1)
        label_quality = 1 if is_autobracket else 2
        
        if deck_tuple in seen_decks:
            # If we already have this exact deck, only overwrite if the new label is higher quality
            if label_quality > seen_decks[deck_tuple]["_quality"]:
                seen_decks[deck_tuple] = {
                    "deck_id": deck_id,
                    "url": url,
                    "cards": list(deck_tuple),
                    "bracket": bracket,
                    "is_autobracket": is_autobracket,
                    "_quality": label_quality
                }
        else:
            seen_decks[deck_tuple] = {
                "deck_id": deck_id,
                "url": url,
                "cards": list(deck_tuple),
                "bracket": bracket,
                "is_autobracket": is_autobracket,
                "_quality": label_quality
            }

    print("Processing ds1 & ds2 (MTGTop8 cEDH extractions)...")
    for path in [ds1_path, ds2_path]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    source = data.get("source", "").lower()
                    
                    # Only extract cEDH/EDH from these general files; skip unlabeled casuals and 60-card formats
                    if "cedh" in source or "edh" in source:
                        cards = list(data.get("mainboard", {}).keys()) + list(data.get("sideboard", {}).keys())
                        deck_id = data.get("deck_id", "unknown")
                        process_and_add_deck(deck_id, None, cards, bracket=5, is_autobracket=False)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping.")

    print("Processing ds3 (Moxfield User/Auto Brackets)...")
    try:
        with open(ds3_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                user_b = data.get("user_bracket")
                auto_b = data.get("auto_bracket")
                
                if user_b is not None:
                    bracket = int(user_b)
                    is_auto = False
                elif auto_b is not None:
                    bracket = int(auto_b)
                    is_auto = True
                else:
                    continue # Skip if completely unlabelled
                    
                cards = [item["n"] for item in data.get("mainboard", [])]
                cards += [item["n"] for item in data.get("commanders", [])]
                deck_id = data.get("id", "unknown")
                url = f"https://moxfield.com/decks/{deck_id}" if deck_id != "unknown" else None
                
                process_and_add_deck(deck_id, url, cards, bracket, is_auto)
    except FileNotFoundError:
        print(f"Warning: {ds3_path} not found. Skipping.")

    print("Processing ds4 (Modern MTGTop8 cEDH)...")
    try:
        with open(ds4_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                cards = [item["name"] for item in data.get("main", [])]
                cards += [item["name"] for item in data.get("cmds", [])]
                deck_id = data.get("deck_id", "unknown")
                url = data.get("deck_url", None)
                
                process_and_add_deck(deck_id, url, cards, bracket=5, is_autobracket=False)
    except FileNotFoundError:
        print(f"Warning: {ds4_path} not found. Skipping.")

    # Flatten the dict values into a list
    final_decks = list(seen_decks.values())
    
    # Remove the temporary _quality key used for parsing logic
    for deck in final_decks:
        del deck["_quality"]

    print(f"Total labeled EDH decks extracted: {len(final_decks)}")
    
    print("Shuffling regression dataset...")
    random.seed(42)
    random.shuffle(final_decks)
    
    print(f"Writing to {out_path}...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for deck in final_decks:
            f.write(json.dumps(deck) + "\n")
            
    print("Dataset compilation complete! Ready for Neural Network training.")

if __name__ == "__main__":
    compile_regression_dataset()