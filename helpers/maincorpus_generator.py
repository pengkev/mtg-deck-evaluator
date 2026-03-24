"""
Regression Dataset Compiler (Brackets 1-5)

Compiles a unified, deduplicated dataset for training the Ordinal Regression model.
Filters out all 60-card formats. Automatically assigns Bracket 5 to tournament cEDH lists.
Prioritizes user-assigned labels over auto-labels during deduplication.
Preserves card quantities for Quantity-Aware Transformer evaluation.

Datasets:
ds1 = "../data/edh-decks.jsonl" (Extracts only cEDH/MTGTop8)
ds2 = "../data/general-decks.jsonl" (Extracts only cEDH/MTGTop8)
ds3 = "../data/large-moxfield-cEDH/official_harvest.jsonl" (130k Moxfield EDH)
ds4 = "../data/large-mtgtop8-cEDH/mtgtop8_decks.jsonl" (24k MTGTop8 cEDH)
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
oracle_cards_path = "../data/oracle_cards.json"
out_path = "../data/big-corpus/supervised_megacorpus.jsonl"


def normalize_card_name(name: str) -> str:
    """Lowercases the card and isolates the front face to unify formatting."""
    name = name.lower()
    front_face = re.split(r'\s*//?\s*', name)[0]
    return front_face.strip()


def load_valid_oracle_names() -> set[str]:
    """Build a normalized card-name set from oracle cards, including split faces."""
    path = Path(oracle_cards_path)
    if not path.exists():
        raise FileNotFoundError(f"Oracle cards file not found: {oracle_cards_path}")

    payload = json.loads(path.read_text(encoding='utf-8'))
    cards = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(cards, list):
        raise ValueError("Unexpected oracle_cards.json structure: expected list or {data: [...]}.")

    valid_names: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue

        name = card.get("name")
        if isinstance(name, str):
            valid_names.add(normalize_card_name(name))

        faces = card.get("card_faces")
        if isinstance(faces, list):
            for face in faces:
                if not isinstance(face, dict):
                    continue
                face_name = face.get("name")
                if isinstance(face_name, str):
                    valid_names.add(normalize_card_name(face_name))

    return valid_names


def compile_regression_dataset():
    # Maps deck_tuple -> deck_data_dict
    seen_decks = {}
    valid_oracle_names = load_valid_oracle_names()
    
    def process_and_add_deck(deck_id, url, card_dict, bracket, is_autobracket):
        # Normalize keys and sum quantities (in case split cards resolve to the same normalized front-face)
        clean_cards = {}
        for name, qty in card_dict.items():
            norm_name = normalize_card_name(name)
            if norm_name not in valid_oracle_names:
                continue
            clean_cards[norm_name] = clean_cards.get(norm_name, 0) + qty
            
        # Require at least 50 unique cards to be considered a valid EDH deck
        if len(clean_cards) < 50:
            return
            
        # Create a sorted tuple of (card_name, quantity) for stable hashing and deduplication
        deck_tuple = tuple(sorted(clean_cards.items()))
        
        # Quality score: Human/Tournament (2) overrides Auto-heuristic (1)
        label_quality = 1 if is_autobracket else 2
        
        if deck_tuple in seen_decks:
            # If we already have this exact deck, only overwrite if the new label is higher quality
            if label_quality > seen_decks[deck_tuple]["_quality"]:
                seen_decks[deck_tuple] = {
                    "deck_id": deck_id,
                    "url": url,
                    "cards": clean_cards, # Store the dictionary format in the final output
                    "bracket": bracket,
                    "is_autobracket": is_autobracket,
                    "_quality": label_quality
                }
        else:
            seen_decks[deck_tuple] = {
                "deck_id": deck_id,
                "url": url,
                "cards": clean_cards, # Store the dictionary format in the final output
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
                    
                    # Only extract cEDH/EDH from these general files
                    if "cedh" in source or "edh" in source:
                        # Merge mainboard and sideboard dictionaries
                        cards = data.get("mainboard", {}).copy()
                        for k, v in data.get("sideboard", {}).items():
                            cards[k] = cards.get(k, 0) + v
                            
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
                    
                # Build the dictionary of counts
                cards = {}
                for item in data.get("mainboard", []):
                    cards[item["n"]] = cards.get(item["n"], 0) + item.get("q", 1)
                for item in data.get("commanders", []):
                    cards[item["n"]] = cards.get(item["n"], 0) + item.get("q", 1)
                    
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
                
                # Build the dictionary of counts
                cards = {}
                for item in data.get("main", []):
                    cards[item["name"]] = cards.get(item["name"], 0) + item.get("qty", 1)
                for item in data.get("cmds", []):
                    cards[item["name"]] = cards.get(item["name"], 0) + item.get("qty", 1)
                    
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