from gensim.models import Word2Vec
from pathlib import Path
import re

"""
Finding: cards are arranged into packages, not necessarily by function. This is great for identifying high power decks but requires attention or cards like command tower and fetches get put on the same tier as underworld breach
"""

MODEL_PATH = Path("../data/general-item2vec_mtg.model")

model = Word2Vec.load(str(MODEL_PATH))

def normalize_card_name(name: str) -> str:
    """Lowercases the card and isolates the front face to unify formatting."""
    name = name.lower()
    front_face = re.split(r'\s*//?\s*', name)[0]
    return front_face.strip()

for test_card in [normalize_card_name(clean_card) for clean_card in ["Ponder", "Balustrade Spy", "Lightning Bolt", "Swords to Plowshares", "Llanowar Elves", "Sol Ring"]]:
    print("\n--- MODEL SANITY CHECK ---")

    try:
        similar_cards = model.wv.most_similar(test_card, topn=5)
        print(f"Cards most similar to '{test_card}':")
        for card, score in similar_cards:
            print(f"  - {card} (Confidence: {score:.2f})")
    except KeyError:
        print(f"'{test_card}' not found in vocabulary. Did you use the right casing?")
        
