import json
import re
from pathlib import Path

# --- Paths ---
INPUT_GENERAL = Path("../data/general-decks.jsonl")
INPUT_EDH = Path("../data/edh-decks.jsonl")
OUTPUT_MEGACORPUS = Path("../data/megacorpus_clean.jsonl")

# --- Normalization ---
def normalize_card_name(name: str) -> str:
    """
    Lowercases the card and isolates the front face to unify formatting.
    Handles 'CardA/CardB' and 'CardA // CardB'.
    """
    name = name.lower()
    # Split on either '//' or '/' and keep only the first half
    front_face = re.split(r'\s*//?\s*', name)[0]
    return front_face.strip()

# Apply normalization to our stop-words
RAW_BASIC_LANDS = [
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp", 
    "Snow-Covered Mountain", "Snow-Covered Forest"
]
BASIC_LANDS = {normalize_card_name(land) for land in RAW_BASIC_LANDS}


def build_megacorpus():
    files_to_process = [INPUT_GENERAL, INPUT_EDH]
    
    # Ensure output directory exists
    OUTPUT_MEGACORPUS.parent.mkdir(parents=True, exist_ok=True)
    
    total_decks_written = 0
    
    print(f"Building unified megacorpus at: {OUTPUT_MEGACORPUS}")
    
    with open(OUTPUT_MEGACORPUS, 'w', encoding='utf-8') as outfile:
        for filepath in files_to_process:
            print(f"Processing {filepath}...")
            
            # Skip if the file doesn't exist yet
            if not filepath.exists():
                print(f"  Warning: {filepath} not found. Skipping.")
                continue
                
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    
                    # Grab all cards
                    mainboard = list(data.get("mainboard", {}).keys())
                    sideboard = list(data.get("sideboard", {}).keys())
                    all_cards = mainboard + sideboard
                    
                    # Normalize and filter out basic lands
                    clean_deck = []
                    for card in all_cards:
                        norm_card = normalize_card_name(card)
                        if norm_card not in BASIC_LANDS:
                            clean_deck.append(norm_card)
                    
                    # Only save viable decks
                    if len(clean_deck) >= 10:
                        # Write the cleaned list of strings directly to the new file
                        # We also carry over the source label in case we need it later!
                        output_data = {
                            "source": data.get("source", "unknown"),
                            "cards": clean_deck
                        }
                        outfile.write(json.dumps(output_data) + "\n")
                        total_decks_written += 1

    print("="*40)
    print(f"Megacorpus complete!")
    print(f"Total decks written: {total_decks_written:,}")
    print("="*40)

if __name__ == "__main__":
    build_megacorpus()