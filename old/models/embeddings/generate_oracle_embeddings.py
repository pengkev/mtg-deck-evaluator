import json
import logging
import re
import unicodedata
import numpy as np
import torch
import math
from pathlib import Path
from typing import Any, Dict, Tuple
from gensim.models import Word2Vec
from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
INPUT_FILE = Path("../../data/cooccurence/unsupervised_megacorpus.jsonl")
MTGJSON_FILE = Path("../../data/oracle_cards.json") # Your oracle text dict
MODEL_OUTPUT = Path("embedding-models/general-item2vec_mtg.model")
HYBRID_OUTPUT = Path("embedding-models/oracle_embeddings.pt") # Saving as a PyTorch tensor
SEMANTIC_MODEL_DIR = Path("embedding-models/mtg-minilm-mlm")
BASE_SEMANTIC_MODEL = "microsoft/MiniLM-L12-H384-uncased"

EMBEDDING_SIZE = 512

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

def load_semantic_encoder() -> Tuple[Any, Any, torch.device, int]:
    """Loads a fine-tuned MiniLM encoder if available, else falls back to base MiniLM."""
    if SEMANTIC_MODEL_DIR.exists():
        model_name_or_path = str(SEMANTIC_MODEL_DIR)
        print(f"Loading fine-tuned semantic model from {SEMANTIC_MODEL_DIR}...")
    else:
        model_name_or_path = BASE_SEMANTIC_MODEL
        print(f"Fine-tuned model not found. Falling back to {BASE_SEMANTIC_MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    return tokenizer, model, device, hidden_size

def encode_text_mean_pool(text: str, tokenizer, model, device: torch.device, max_length: int = 256) -> np.ndarray:
    """Encodes text with transformer mean pooling over non-padding tokens."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        sentence_embedding = summed / counts

    return sentence_embedding.squeeze(0).detach().cpu().numpy()

def normalize_card_name(name: str) -> str:
    """Normalizes names for robust vocab <-> oracle matching."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"^a-", "", name)
    name = re.sub(r"\s+", " ", name)
    front_face = re.split(r'\s*//\s*', name)[0]
    return front_face.strip()

def build_card_metadata_header(card: Dict[str, Any]) -> str:
    """Builds a semantic description of the card's physical properties."""
    chunks = []

    name = (card.get("name") or "").strip()
    mana_cost = (card.get("mana_cost") or "").strip()
    type_line = (card.get("type_line") or "").strip()
    power = (card.get("power") or "").strip()
    toughness = (card.get("toughness") or "").strip()
    loyalty = (card.get("loyalty") or "").strip()

    if name:
        chunks.append(f"Name: {name}.")
    if mana_cost:
        chunks.append(f"Mana cost: {mana_cost}.")
    if type_line:
        chunks.append(f"Type: {type_line}.")
    if power or toughness:
        chunks.append(f"Stats: {power}/{toughness}.")
    if loyalty:
        chunks.append(f"Loyalty: {loyalty}.")

    for face in card.get("card_faces", []):
        face_name = (face.get("name") or "").strip()
        face_mana = (face.get("mana_cost") or "").strip()
        face_type = (face.get("type_line") or "").strip()
        face_pt = ""
        if face.get("power") or face.get("toughness"):
            face_pt = f" {face.get('power', '')}/{face.get('toughness', '')}".rstrip()

        face_parts = [p for p in [face_name, face_mana, face_type] if p]
        if face_parts:
            summary = " | ".join(face_parts)
            if face_pt:
                summary += f" | Stats:{face_pt}"
            chunks.append(f"Face: {summary}.")

    return " ".join(chunks).strip()

def get_combined_oracle_text(card: Dict[str, Any]) -> str:
    """Returns combined physical metadata AND Oracle text for all cards."""
    parts = []

    # 1. Always prepend the card's physical metadata header.
    header_text = build_card_metadata_header(card)
    if header_text:
        parts.append(header_text)

    # 2. Append top-level Oracle text.
    top_level_text = (card.get("oracle_text") or "").strip()
    if top_level_text:
        parts.append(top_level_text)

    # 3. Append multi-face Oracle text.
    for face in card.get("card_faces", []):
        face_text = (face.get("oracle_text") or "").strip()
        if face_text:
            parts.append(face_text)

    # De-duplicate identical text blocks.
    deduped_parts = list(dict.fromkeys(parts))
    return "\n".join(deduped_parts)

def build_oracle_index(json_filepath: Path) -> Dict[str, str]:
    """
    Builds an index of normalized card name -> combined Oracle text.
    Works for normal cards and split/MDFC/modal cards.
    """
    print(f"Parsing Oracle text from {json_filepath}...")
    with open(json_filepath, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)

    oracle_index: Dict[str, str] = {}

    for card in cards_data:
        candidate_names = [card.get("name", "")]
        candidate_names.extend(face.get("name", "") for face in card.get("card_faces", []))

        normalized_names = {normalize_card_name(name) for name in candidate_names if name}
        normalized_names = {name for name in normalized_names if name}
        if not normalized_names:
            continue

        combined_text = get_combined_oracle_text(card)

        # Index by main card name and each face name to maximize lookup hit-rate.
        for norm_name in normalized_names:
            existing_text = oracle_index.get(norm_name, "")
            if not existing_text.strip() and combined_text.strip():
                oracle_index[norm_name] = combined_text
            elif norm_name not in oracle_index:
                oracle_index[norm_name] = combined_text

    print(f"Successfully extracted Oracle text for {len(oracle_index)} unique cards.")
    return oracle_index

class MTGDeckCorpus:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip(): continue
                data = json.loads(line)
                if "cards" in data:
                    yield data["cards"]

def train_w2v_embeddings():
    print(f"Initializing stream from {INPUT_FILE}...")
    sentences = MTGDeckCorpus(INPUT_FILE)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    print("Training Gensim Word2Vec Model...")
    model = Word2Vec(
        sentences=sentences, 
        vector_size=EMBEDDING_SIZE,  
        window=115,       
        min_count=3,      
        sg=1,             
        workers=12,       
        epochs=10         
    )
    
    model.save(str(MODEL_OUTPUT))
    return model

def build_hybrid_embeddings(w2v_model, mtgjson_path, drop_missing_oracle=True):
    print("\n--- BUILDING HYBRID (W2V + ORACLE) EMBEDDINGS ---")
    
    # 1. Build normalized-name Oracle index from oracle_cards.json
    oracle_index = build_oracle_index(mtgjson_path)
        
    # 2. Load semantic model (fine-tuned MiniLM preferred)
    tokenizer, nlp_model, nlp_device, nlp_dim = load_semantic_encoder()
    
    vocab = w2v_model.wv.index_to_key

    if drop_missing_oracle:
        filtered_vocab = [
            card_name
            for card_name in vocab
            if oracle_index.get(normalize_card_name(card_name), "").strip()
        ]
        dropped_count = len(vocab) - len(filtered_vocab)
        print(f"Dropping {dropped_count} cards with empty indexed oracle text.")
    else:
        filtered_vocab = list(vocab)
        dropped_count = 0

    w2v_dim = w2v_model.vector_size
    
    # We add <PAD> (0) and <UNK> (1) to the final matrix
    vocab_size = len(filtered_vocab) + 2
    hybrid_dim = w2v_dim + nlp_dim
    
    hybrid_matrix = np.zeros((vocab_size, hybrid_dim))
    final_vocab_dict = {"<PAD>": 0, "<UNK>": 1}
    
    # Track missing oracle text for diagnostics
    missing_oracle_count = 0
    text_embedding_cache: Dict[str, np.ndarray] = {}
    
    print("Fusing vectors...")
    for i, card_name in enumerate(filtered_vocab):
        pytorch_id = i + 2
        final_vocab_dict[card_name] = pytorch_id
        
        # --- Fusion Weights ---
        # 0.5 means exactly equal influence. 
        # 0.7 means 70% Word2Vec (Meta), 30% NLP (Mechanics)
        W2V_WEIGHT = 0.4 
        NLP_WEIGHT = 0.6 
        
        # --- Word2Vec (Co-occurrence) ---
        vec_w2v = w2v_model.wv[card_name]
        norm_w2v = np.linalg.norm(vec_w2v)
        if norm_w2v > 0: 
            # Normalize to 1, then scale by the square root of its weight
            vec_w2v = (vec_w2v / norm_w2v) * math.sqrt(W2V_WEIGHT)
            
        # --- Oracle Text (Semantics) ---
        oracle_text = oracle_index.get(normalize_card_name(card_name), "")
        
        if not oracle_text:
            missing_oracle_count += 1
            vec_nlp = np.zeros(nlp_dim)
        else:
            if oracle_text in text_embedding_cache:
                vec_nlp = text_embedding_cache[oracle_text]
            else:
                vec_nlp = encode_text_mean_pool(oracle_text, tokenizer, nlp_model, nlp_device)
                text_embedding_cache[oracle_text] = vec_nlp
            norm_nlp = np.linalg.norm(vec_nlp)
            if norm_nlp > 0: 
                # Normalize to 1, then scale by the square root of its weight
                vec_nlp = (vec_nlp / norm_nlp) * math.sqrt(NLP_WEIGHT)
                
        # --- Concatenate ---
        fused_vector = np.concatenate([vec_w2v, vec_nlp])
        
        # (Optional but good practice) Final safety normalization to guarantee exactly 1.0
        norm_fused = np.linalg.norm(fused_vector)
        if norm_fused > 0:
            fused_vector = fused_vector / norm_fused
            
        hybrid_matrix[pytorch_id] = fused_vector
        
    print(f"Fusion complete. Cards missing Oracle text: {missing_oracle_count}")
    if drop_missing_oracle:
        print(f"Total cards removed from output vocab: {dropped_count}")
    
    # Save the final matrix and vocab dictionary for PyTorch
    torch.save({
        'weights': torch.FloatTensor(hybrid_matrix),
        'vocab': final_vocab_dict
    }, HYBRID_OUTPUT)
    
    print(f"Saved Hybrid Tensor ({vocab_size}x{hybrid_dim}) to {HYBRID_OUTPUT}")
    return final_vocab_dict, hybrid_matrix

if __name__ == "__main__":
    # 1. Train or load W2V
    if not MODEL_OUTPUT.exists():
        trained_model = train_w2v_embeddings()
    else:
        print("Loading existing W2V model...")
        trained_model = Word2Vec.load(str(MODEL_OUTPUT))
        
    # 2. Build Hybrid Embeddings
    vocab_dict, hybrid_weights = build_hybrid_embeddings(trained_model, MTGJSON_FILE)