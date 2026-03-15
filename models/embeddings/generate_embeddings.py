"""
Standalone Word2Vec Embedding Generator for Magic: The Gathering
Reads from a pre-compiled, shuffled JSONL corpus and trains a Gensim model.
"""

import json
import logging
from pathlib import Path
from gensim.models import Word2Vec

# --- Configuration ---
INPUT_FILE = Path("../../data/embeddings/unsupervised_megacorpus.jsonl")
MODEL_OUTPUT = Path("../data/general-item2vec_mtg.model")
EMBEDDING_SIZE = 512

# Basic logging to watch Gensim train in the console
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class MTGDeckCorpus:
    """
    An ultra-fast iterable that streams pre-cleaned decks directly into Gensim.
    Since the data was already filtered and shuffled, we just yield the arrays.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                # The compilation script saved the lists under the "cards" key
                if "cards" in data:
                    yield data["cards"]

def train_embeddings():
    print(f"Initializing stream from {INPUT_FILE}...")
    sentences = MTGDeckCorpus(INPUT_FILE)
    
    # Ensure output directory exists
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    print("Training Gensim Word2Vec Model...")
    model = Word2Vec(
        sentences=sentences, 
        vector_size=EMBEDDING_SIZE,  
        window=115,       # "Infinite Window" covers the whole deck
        min_count=3,      # Ignore incredibly obscure/custom cards
        sg=1,             # 1 = Skip-Gram (Item2Vec)
        workers=12,       # CPU cores to use
        epochs=10         # Passes over the dataset
    )
    
    print(f"Training complete! Saving model to {MODEL_OUTPUT}")
    model.save(str(MODEL_OUTPUT))
    return model

def run_sanity_check(model):
    print("\n" + "="*40)
    print("--- EMBEDDINGS SANITY CHECK ---")
    print("="*40)
    
    # Using normalized names since the corpus was pre-normalized
    test_cards = [
        "ponder", 
        "demonic tutor", 
        "lightning bolt", 
        "swords to plowshares", 
        "llanowar elves", 
        "sol ring"
    ]
    
    for card in test_cards:
        try:
            similar_cards = model.wv.most_similar(card, topn=5)
            print(f"\nCards most similar to '{card}':")
            for sim_card, score in similar_cards:
                print(f"  - {sim_card} (Confidence: {score:.2f})")
        except KeyError:
            print(f"\n'{card}' not found in vocabulary. Check preprocessing.")

if __name__ == "__main__":
    # 1. Train the model
    trained_model = train_embeddings()
    
    # 2. Verify it learned the correct package clusters
    run_sanity_check(trained_model)