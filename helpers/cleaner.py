import json
import re
from pathlib import Path
from collections import defaultdict

# --- Paths ---
INPUT_GENERAL = Path("../data/general-decks.jsonl")
INPUT_EDH = Path("../data/edh-decks.jsonl")

# We save them as "-clean" so we don't accidentally overwrite the originals if something goes wrong
OUTPUT_GENERAL = Path("../data/clean-general-decks.jsonl")
OUTPUT_EDH = Path("../data/clean-edh-decks.jsonl")

FILES_TO_PROCESS = [
    (INPUT_GENERAL, OUTPUT_GENERAL),
    (INPUT_EDH, OUTPUT_EDH)
]

# --- Normalization ---
def normalize_card_name(name: str) -> str:
    """
    Lowercases the card and isolates the front face to unify formatting.
    Handles 'CardA/CardB' and 'CardA // CardB'.
    """
    name = name.lower()
    front_face = re.split(r'\s*//?\s*', name)[0]
    return front_face.strip()

RAW_BASIC_LANDS = [
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp", 
    "Snow-Covered Mountain", "Snow-Covered Forest"
]
BASIC_LANDS = {normalize_card_name(land) for land in RAW_BASIC_LANDS}

def clean_board(board_dict: dict) -> dict:
    """
    Takes a dictionary of {card_name: quantity}.
    Returns a new dict with normalized names, filtered basic lands, and summed quantities.
    """
    clean_dict = defaultdict(int)
    for card, count in board_dict.items():
        norm_card = normalize_card_name(card)
        if norm_card not in BASIC_LANDS:
            clean_dict[norm_card] += count
            
    # Convert back to standard dict for clean JSON serialization
    return dict(clean_dict)

def clean_jsonls():
    print("Starting JSONL text normalization...\n")
    
    for input_path, output_path in FILES_TO_PROCESS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cleaning: {input_path.name}")
        print(f"Saving to: {output_path.name}")
        
        if not input_path.exists():
            print(f"  [!] Warning: {input_path} not found. Skipping.\n")
            continue
            
        decks_written = 0
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                # Clean the dictionaries
                clean_main = clean_board(data.get("mainboard", {}))
                clean_side = clean_board(data.get("sideboard", {}))
                
                # We count unique cards (keys) to match our previous viable deck threshold
                unique_cards = len(clean_main) + len(clean_side)
                
                if unique_cards >= 10:
                    # Overwrite the old dictionaries with the cleaned ones
                    data["mainboard"] = clean_main
                    data["sideboard"] = clean_side
                    
                    # Write the fully intact (but clean) JSON object back to the new file
                    outfile.write(json.dumps(data) + "\n")
                    decks_written += 1
                    
        print(f"  -> Done! Wrote {decks_written:,} clean decks.\n")

if __name__ == "__main__":
    clean_jsonls()