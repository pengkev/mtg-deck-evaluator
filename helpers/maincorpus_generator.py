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

ds3 schema: {"id": "jUdbHQ9EAE-avR32TxuFPA", "name": "Sisay (cEDH)", "user_bracket": 5, "auto_bracket": 4, "hubs": [], "mainboard": [{"n": "Smothering Tithe", "q": 1}, {"n": "Mana Vault", "q": 1}, {"n": "Misty Rainforest", "q": 1}, {"n": "Pact of Negation", "q": 1}, {"n": "Silence", "q": 1}, {"n": "Noble Hierarch", "q": 1}, {"n": "Ancient Tomb", "q": 1}, {"n": "Crop Rotation", "q": 1}, {"n": "Teferi, Time Raveler", "q": 1}, {"n": "Enlightened Tutor", "q": 1}, {"n": "Fierce Guardianship", "q": 1}, {"n": "City of Brass", "q": 1}, {"n": "Mana Confluence", "q": 1}, {"n": "Abrupt Decay", "q": 1}, {"n": "Deathrite Shaman", "q": 1}, {"n": "Mental Misstep", "q": 1}, {"n": "Gaea's Cradle", "q": 1}, {"n": "Bayou", "q": 1}, {"n": "Plateau", "q": 1}, {"n": "Scrubland", "q": 1}, {"n": "Savannah", "q": 1}, {"n": "Tropical Island", "q": 1}, {"n": "Tundra", "q": 1}, {"n": "Taiga", "q": 1}, {"n": "Volcanic Island", "q": 1}, {"n": "Kinnan, Bonder Prodigy", "q": 1}, {"n": "Underground Sea", "q": 1}, {"n": "Elvish Spirit Guide", "q": 1}, {"n": "Mox Diamond", "q": 1}, {"n": "Flooded Strand", "q": 1}, {"n": "Polluted Delta", "q": 1}, {"n": "Windswept Heath", "q": 1}, {"n": "Wooded Foothills", "q": 1}, {"n": "Gemstone Caverns", "q": 1}, {"n": "Saheeli Rai", "q": 1}, {"n": "Sol Ring", "q": 1}, {"n": "Vampiric Tutor", "q": 1}, {"n": "Mox Amber", "q": 1}, {"n": "Mindbreak Trap", "q": 1}, {"n": "Bloom Tender", "q": 1}, {"n": "Simian Spirit Guide", "q": 1}, {"n": "Chrome Mox", "q": 1}, {"n": "Tarnished Citadel", "q": 1}, {"n": "Command Tower", "q": 1}, {"n": "Marsh Flats", "q": 1}, {"n": "Scalding Tarn", "q": 1}, {"n": "Verdant Catacombs", "q": 1}, {"n": "Esper Sentinel", "q": 1}, {"n": "Ignoble Hierarch", "q": 1}, {"n": "Exotic Orchard", "q": 1}, {"n": "Otawara, Soaring City", "q": 1}, {"n": "Touch the Spirit Realm", "q": 1}, {"n": "Colossal Skyturtle", "q": 1}, {"n": "Boseiju, Who Endures", "q": 1}, {"n": "Green Sun's Zenith", "q": 1}, {"n": "Emiel the Blessed", "q": 1}, {"n": "Force of Will", "q": 1}, {"n": "Relic of Legends", "q": 1}, {"n": "Ertai Resurrected", "q": 1}, {"n": "Birds of Paradise", "q": 1}, {"n": "Rhystic Study", "q": 1}, {"n": "Mystic Remora", "q": 1}, {"n": "Tyvar, Jubilant Brawler", "q": 1}, {"n": "Sakashima the Impostor", "q": 1}, {"n": "Mount Doom", "q": 1}, {"n": "Drana and Linvala", "q": 1}, {"n": "Selvala, Heart of the Wilds", "q": 1}, {"n": "Esika, God of the Tree // The Prismatic Bridge", "q": 1}, {"n": "Orcish Bowmasters", "q": 1}, {"n": "Swan Song", "q": 1}, {"n": "Demonic Tutor", "q": 1}, {"n": "Lotus Petal", "q": 1}, {"n": "Agatha's Soul Cauldron", "q": 1}, {"n": "Faeburrow Elder", "q": 1}, {"n": "Legolas's Quick Reflexes", "q": 1}, {"n": "Lotho, Corrupt Shirriff", "q": 1}, {"n": "Ioreth of the Healing House", "q": 1}, {"n": "Delighted Halfling", "q": 1}, {"n": "Kutzil, Malamet Exemplar", "q": 1}, {"n": "Arcane Signet", "q": 1}, {"n": "Ruby, Daring Tracker", "q": 1}, {"n": "Flusterstorm", "q": 1}, {"n": "Ragavan, Nimble Pilferer", "q": 1}, {"n": "Basim Ibn Ishaq", "q": 1}, {"n": "Into the Flood Maw", "q": 1}, {"n": "Derevi, Empyrial Tactician", "q": 1}, {"n": "Fellwar Stone", "q": 1}, {"n": "Lavinia, Azorius Renegade", "q": 1}, {"n": "Marvin, Murderous Mimic", "q": 1}, {"n": "Enduring Vitality", "q": 1}, {"n": "Deflecting Swat", "q": 1}, {"n": "Nature's Rhythm", "q": 1}, {"n": "Dismember", "q": 1}, {"n": "Deadpool, Trading Card", "q": 1}, {"n": "Tataru Taru", "q": 1}, {"n": "The Cabbage Merchant", "q": 1}, {"n": "Wan Shi Tong, Librarian", "q": 1}, {"n": "Badgermole Cub", "q": 1}, {"n": "Tam, Mindful First-Year", "q": 1}], "commanders": [{"n": "Sisay, Weatherlight Captain", "q": 1}]}
ds4 schema: {"deck_id": "818305", "deck_url": "https://www.mtgtop8.com/event?e=81365&d=818305&f=cEDH", "mtgo_url": "https://www.mtgtop8.com/mtgo/event?e=81365&d=818305&f=cEDH", "date": "02/03/26", "placement": 5, "players": 25, "placement_of": "5/25", "main": [{"name": "Archivist of Oghma", "qty": 1}, {"name": "Aven Interrupter", "qty": 1}, {"name": "Birds of Paradise", "qty": 1}, {"name": "Dauthi Voidwalker", "qty": 1}, {"name": "Deathrite Shaman", "qty": 1}, {"name": "Delighted Halfling", "qty": 1}, {"name": "Drana and Linvala", "qty": 1}, {"name": "Drannith Magistrate", "qty": 1}, {"name": "Elesh Norn, Grand Cenobite", "qty": 1}, {"name": "Endurance", "qty": 1}, {"name": "Enduring Vitality", "qty": 1}, {"name": "Esper Sentinel", "qty": 1}, {"name": "Felidar Guardian", "qty": 1}, {"name": "Goblin Sharpshooter", "qty": 1}, {"name": "Grim Hireling", "qty": 1}, {"name": "Haywire Mite", "qty": 1}, {"name": "Hazel's Brewmaster", "qty": 1}, {"name": "Hexing Squelcher", "qty": 1}, {"name": "Ignoble Hierarch", "qty": 1}, {"name": "Karmic Guide", "qty": 1}, {"name": "Kiki-Jiki, Mirror Breaker", "qty": 1}, {"name": "Knuckles the Echidna", "qty": 1}, {"name": "Kutzil, Malamet Exemplar", "qty": 1}, {"name": "Lotho, Corrupt Shirriff", "qty": 1}, {"name": "Mayhem Devil", "qty": 1}, {"name": "Opposition Agent", "qty": 1}, {"name": "Orcish Bowmasters", "qty": 1}, {"name": "Ranger-Captain of Eos", "qty": 1}, {"name": "Rev, Tithe Extractor", "qty": 1}, {"name": "Shalai, Voice of Plenty", "qty": 1}, {"name": "Sire of Insanity", "qty": 1}, {"name": "Solitude", "qty": 1}, {"name": "The Cabbage Merchant", "qty": 1}, {"name": "The Jolly Balloon Man", "qty": 1}, {"name": "Village Bell-Ringer", "qty": 1}, {"name": "Voice of Victory", "qty": 1}, {"name": "Wandering Archaic", "qty": 1}, {"name": "Birthing Pod", "qty": 1}, {"name": "Chrome Mox", "qty": 1}, {"name": "Mox Diamond", "qty": 1}, {"name": "Sol Ring", "qty": 1}, {"name": "The One Ring", "qty": 1}, {"name": "Abrupt Decay", "qty": 1}, {"name": "Assassin's Trophy", "qty": 1}, {"name": "Chord of Calling", "qty": 1}, {"name": "Crop Rotation", "qty": 1}, {"name": "Deflecting Swat", "qty": 1}, {"name": "Eladamri's Call", "qty": 1}, {"name": "Enlightened Tutor", "qty": 1}, {"name": "Fire Covenant", "qty": 1}, {"name": "Force of Vigor", "qty": 1}, {"name": "Legolas's Quick Reflexes", "qty": 1}, {"name": "Red Elemental Blast", "qty": 1}, {"name": "Silence", "qty": 1}, {"name": "Swords to Plowshares", "qty": 1}, {"name": "Vampiric Tutor", "qty": 1}, {"name": "Veil of Summer", "qty": 1}, {"name": "Worldly Tutor", "qty": 1}, {"name": "Culling Ritual", "qty": 1}, {"name": "Demonic Tutor", "qty": 1}, {"name": "Diabolic Intent", "qty": 1}, {"name": "Eldritch Evolution", "qty": 1}, {"name": "Finale of Devastation", "qty": 1}, {"name": "Imperial Seal", "qty": 1}, {"name": "Nature's Rhythm", "qty": 1}, {"name": "Splinter Twin", "qty": 1}, {"name": "Survival of the Fittest", "qty": 1}, {"name": "Vivien on the Hunt", "qty": 1}, {"name": "Ancient Tomb", "qty": 1}, {"name": "Arid Mesa", "qty": 1}, {"name": "Badlands", "qty": 1}, {"name": "Bayou", "qty": 1}, {"name": "Blood Crypt", "qty": 1}, {"name": "Bloodstained Mire", "qty": 1}, {"name": "Boseiju, Who Endures", "qty": 1}, {"name": "City of Brass", "qty": 1}, {"name": "Command Tower", "qty": 1}, {"name": "Flooded Strand", "qty": 1}, {"name": "Gaea's Cradle", "qty": 1}, {"name": "Gemstone Caverns", "qty": 1}, {"name": "Godless Shrine", "qty": 1}, {"name": "Mana Confluence", "qty": 1}, {"name": "Marsh Flats", "qty": 1}, {"name": "Misty Rainforest", "qty": 1}, {"name": "Overgrown Tomb", "qty": 1}, {"name": "Plateau", "qty": 1}, {"name": "Polluted Delta", "qty": 1}, {"name": "Savannah", "qty": 1}, {"name": "Scalding Tarn", "qty": 1}, {"name": "Scrubland", "qty": 1}, {"name": "Stomping Ground", "qty": 1}, {"name": "Taiga", "qty": 1}, {"name": "Talon Gates of Madara", "qty": 1}, {"name": "Tarnished Citadel", "qty": 1}, {"name": "Temple Garden", "qty": 1}, {"name": "Verdant Catacombs", "qty": 1}, {"name": "Windswept Heath", "qty": 1}, {"name": "Wooded Foothills", "qty": 1}], "cmds": [{"name": "Tana, the Bloodsower", "qty": 1}, {"name": "Tymna the Weaver", "qty": 1}]}

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
    
    def process_and_add_deck(deck_id, url, card_dict, commander_dict, bracket, is_autobracket):
        # Normalize keys and sum quantities (in case split cards resolve to the same normalized front-face)
        clean_cards = {}
        for name, qty in card_dict.items():
            norm_name = normalize_card_name(name)
            if norm_name not in valid_oracle_names:
                continue
            clean_cards[norm_name] = clean_cards.get(norm_name, 0) + qty

        clean_commanders = {}
        for name, qty in commander_dict.items():
            norm_name = normalize_card_name(name)
            if norm_name not in valid_oracle_names:
                continue
            clean_commanders[norm_name] = clean_commanders.get(norm_name, 0) + qty
            
        # Require at least 50 unique cards to be considered a valid EDH deck
        unique_deck_cards = set(clean_cards) | set(clean_commanders)
        if len(unique_deck_cards) < 50:
            return
            
        # Create sorted tuples for stable hashing and deduplication, preserving commander identity
        deck_tuple = (
            tuple(sorted(clean_cards.items())),
            tuple(sorted(clean_commanders.items()))
        )
        
        # Quality score: Human/Tournament (2) overrides Auto-heuristic (1)
        label_quality = 1 if is_autobracket else 2
        
        if deck_tuple in seen_decks:
            # If we already have this exact deck, only overwrite if the new label is higher quality
            if label_quality > seen_decks[deck_tuple]["_quality"]:
                seen_decks[deck_tuple] = {
                    "deck_id": deck_id,
                    "url": url,
                    "cards": clean_cards, # Store the dictionary format in the final output
                    "commanders": clean_commanders,
                    "bracket": bracket,
                    "is_autobracket": is_autobracket,
                    "_quality": label_quality
                }
        else:
            seen_decks[deck_tuple] = {
                "deck_id": deck_id,
                "url": url,
                "cards": clean_cards, # Store the dictionary format in the final output
                "commanders": clean_commanders,
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
                        process_and_add_deck(deck_id, None, cards, {}, bracket=5, is_autobracket=False)
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

                commanders = {}
                for item in data.get("commanders", []):
                    commanders[item["n"]] = commanders.get(item["n"], 0) + item.get("q", 1)
                    
                deck_id = data.get("id", "unknown")
                url = f"https://moxfield.com/decks/{deck_id}" if deck_id != "unknown" else None
                
                process_and_add_deck(deck_id, url, cards, commanders, bracket, is_auto)
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

                commanders = {}
                for item in data.get("cmds", []):
                    commanders[item["name"]] = commanders.get(item["name"], 0) + item.get("qty", 1)
                    
                deck_id = data.get("deck_id", "unknown")
                url = data.get("deck_url", None)
                
                process_and_add_deck(deck_id, url, cards, commanders, bracket=5, is_autobracket=False)
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