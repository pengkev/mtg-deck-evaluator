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
out_path = "../data/embedding-corpus/unsupervised_megacorpus.jsonl"

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

def compile_megacorpus():
    seen_decks = set()
    all_unique_decks = []
    valid_oracle_names = load_valid_oracle_names()
    
    def process_and_add_deck(card_names):
        # Normalize and remove duplicate cards within the same deck
        # This reduces 15x Island to 1x Island, preventing context window flooding in Word2Vec
        clean_cards = {
            norm_name
            for c in card_names
            for norm_name in [normalize_card_name(c)]
            if norm_name in valid_oracle_names
        }
        
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