"""
MTG Deck Power Level Classifier - Inference Script
Runs a raw decklist through the pre-trained Deep Sets Attention model.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from gensim.models import Word2Vec

# --- CONFIGURATION ---
EMBEDDING_MODEL_PATH = Path("../embeddings/embedding-models/512dim_item2vec_mtg.model")
PYTORCH_WEIGHTS_PATH = Path("lr0.001_h128_model_weights.pth") # Update this to your best saved model
EMBEDDING_SIZE = 512
HIDDEN_DIM = 128
MAX_LEN = 115

RAW_BASIC_LANDS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp", 
    "Snow-Covered Mountain", "Snow-Covered Forest"
}

def normalize_card_name(name: str) -> str:
    name = name.lower()
    front_face = re.split(r'\s*//?\s*', name)[0]
    return front_face.strip()

BASIC_LANDS = {normalize_card_name(land) for land in RAW_BASIC_LANDS}

# --- MODEL ARCHITECTURE ---
class EDHAttentionDeepSets(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_SIZE, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
        self.card_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.deck_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 5) # 5 Classes for Brackets 1-5
        )
        
    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1) 
        embedded = self.embedding(x)
        card_features = self.card_mlp(embedded) 
        
        attn_scores = self.attention_net(card_features) 
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1) 
        
        deck_vector = torch.sum(attn_weights * card_features, dim=1)
        return self.deck_mlp(deck_vector), attn_weights

# --- HELPER FUNCTIONS ---
def load_gensim_vocab(gensim_model_path):
    """Loads just the vocabulary mapping from the Gensim model."""
    print(f"Loading Gensim vocabulary from {gensim_model_path}...")
    g_model = Word2Vec.load(str(gensim_model_path))
    gensim_words = g_model.wv.index_to_key 
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(gensim_words):
        vocab[word] = i + 2 
    return vocab

def parse_decklist(raw_text, vocab):
    """Parses a text decklist into a padded tensor of vocabulary IDs."""
    deck_cards = []
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line or line.lower() == 'sideboard': 
            continue
        
        match = re.match(r'^\d+\s+(.+)$', line)
        if match:
            norm_card = normalize_card_name(match.group(1))
            if norm_card not in BASIC_LANDS:
                deck_cards.append(norm_card)

    deck_ids = [vocab.get(card, 1) for card in deck_cards] # 1 is <UNK>
    
    if len(deck_ids) > MAX_LEN:
        deck_ids = deck_ids[:MAX_LEN]
    else:
        deck_ids += [0] * (MAX_LEN - len(deck_ids)) # 0 is <PAD>
        
    print(f"Parsed {len(deck_cards)} non-basic cards.")
    return torch.tensor(deck_ids, dtype=torch.long).unsqueeze(0), deck_cards

def run_inference_and_visualize(model, deck_tensor, card_names, vocab, top_n=12):
    """Runs the model and plots the attention weights."""
    model.eval()
    device = next(model.parameters()).device
    deck_tensor = deck_tensor.to(device)
    inv_vocab = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        logits, attn_weights = model(deck_tensor)
        # Convert 0-indexed output back to 1-indexed Bracket
        pred_label = torch.argmax(logits, dim=1).item() + 1
        confidence = F.softmax(logits, dim=1)[0][pred_label - 1].item() * 100

    # Clean up weights for plotting
    weights_np = attn_weights.squeeze().cpu().numpy()
    ids_np = deck_tensor.squeeze().cpu().numpy()
    
    valid_indices = (ids_np != 0) & (ids_np != 1)
    clean_ids = ids_np[valid_indices]
    clean_weights = weights_np[valid_indices]
    
    clean_card_names = [inv_vocab.get(i, f"ID:{i}") for i in clean_ids]

    # Sort and slice top N
    sorted_indices = np.argsort(clean_weights)[::-1][:top_n]
    top_cards = [clean_card_names[i] for i in sorted_indices]
    top_weights = [clean_weights[i] for i in sorted_indices]

    # Plot 
    plt.figure(figsize=(10, 6))
    plt.barh(top_cards, top_weights, color='teal')
    plt.xlabel("Attention Weight (Importance to Classification)")
    plt.title(f"Predicted: Bracket {pred_label} ({confidence:.1f}% Confidence)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Vocab
    vocab_dict = load_gensim_vocab(EMBEDDING_MODEL_PATH)
    
    # 2. Initialize Model and Load Weights
    print(f"Loading PyTorch weights from {PYTORCH_WEIGHTS_PATH}...")
    model = EDHAttentionDeepSets(vocab_size=len(vocab_dict), embedding_dim=EMBEDDING_SIZE, hidden_dim=HIDDEN_DIM)
    
    # Load state dict (mapping CPU/GPU appropriately)
    model.load_state_dict(torch.load(PYTORCH_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)
    
    # 3. Input Decklist
    test_deck = """
    1 Arcane Signet
    1 Aura Shards
    1 Beast Whisperer
    1 Beast Within
    1 Birds of Paradise
    1 Bountiful Promenade
    1 Bruvac the Grandiloquent
    1 Cavern of Souls
    1 Chulane, Teller of Tales
    1 Command Tower
    1 Cyclonic Rift
    1 Deserted Beach
    1 Disallow
    1 Dreamroot Cascade
    1 Drumbellower
    1 Dusk // Dawn
    1 Flooded Grove
    1 Flooded Strand
    4 Forest
    1 Frilled Mystic
    1 Generous Gift
    1 Grand Abolisher
    1 Grand Arbiter Augustin IV
    1 Herald's Horn
    6 Island
    1 Jeweled Lotus
    1 Kinnan, Bonder Prodigy
    1 Laboratory Maniac
    1 Lotus Cobra
    1 Maddening Cacophony
    1 Misty Rainforest
    1 Mystic Gate
    1 Mystic Snake
    1 Noble Hierarch
    1 Overgrown Farmland
    28 Persistent Petitioners
    4 Plains
    1 Quest for Renewal
    1 Rally the Ancestors
    1 Ranger-Captain of Eos
    1 Rejuvenating Springs
    1 Rhystic Study
    1 Sakura-Tribe Elder
    1 Sapphire Medallion
    1 Savannah
    1 Sea of Clouds
    1 Seedborn Muse
    1 Shrieking Drake
    1 Skycloud Expanse
    1 Sol Ring
    1 Sungrass Prairie
    1 Swords to Plowshares
    1 Tatyova, Benthic Druid
    1 Teferi's Protection
    1 Thrumming Stone
    1 Tireless Provisioner
    1 Tropical Island
    1 Tundra
    1 Village Bell-Ringer
    1 Voidslime
    1 Windswept Heath
    1 Wooded Bastion
    """
    
    # 4. Parse and Predict
    deck_tensor, parsed_names = parse_decklist(test_deck, vocab_dict)
    run_inference_and_visualize(model, deck_tensor, parsed_names, vocab_dict)