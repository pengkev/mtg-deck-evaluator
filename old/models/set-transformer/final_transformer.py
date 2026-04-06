# %% [markdown]
# # Classifying MTG Decks with Set Transformers
# 

# %% [markdown]
# This notebook builds a Set Transformer that scores EDH deck power by combining strong static card embeddings with context-aware attention. The model is trained in three phases: EDH-only masked language modeling for synergy learning, supervised bracket regression for global calibration, and tournament ranking for competitive top-end alignment.

# %% [markdown]
# # 0. Setup

# %% [markdown]
# This notebook supports end-to-end training, diagnostics, and checkpointing. Section 0 defines corpus selection, model geometry, and training controls so runs are reproducible and architecture intent is explicit.

# %%
import json
import math
import random
import re
import unicodedata
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# %% [markdown]
# This project uses three corpora with distinct roles. The unsupervised multi-format corpus is used for optional Word2Vec co-occurrence embedding generation only. The supervised Moxfield + MTGTop8 EDH corpus is the core training source for Phase 1 MLM and Phase 2 regression so the model's deck-language prior stays EDH-specific. The tournament plus Game Knights corpus is used for Phase 3 pairwise ranking, with softer confidence on Game Knights outcomes to account for non-pure decklist variance, such as piloting and politics, which are more relevant at lower levels.

# %%
UNSUPERVISED_JSONL = Path("../../data/cooccurence/embedding_corpus.jsonl")
SUPERVISED_JSONL = Path("../../data/supervised/supervised_corpus.jsonl")
TOURNAMENT_JSONL = Path("../../data/tournament/tournament_corpus.jsonl")
GAMEKNIGHTS_JSONL = Path("../../data/gameknights_archidekt_decks.jsonl")
COMBINED_TOURNAMENT_JSONL = Path("../../data/tournament/combined_tournament_plus_gameknights.jsonl")

# %% [markdown]
# For preloaded embeddings, generated with embeddings model, and saving model checkpoints for later training

# %%
HYBRID_EMBED_PATH = Path("../embeddings/embedding-models/896dim_oracle_embeddings.pt")
CHECKPOINT_DIR = Path("./checkpoints")

# Core special-token conventions
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"

def normalize_card_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"^a-", "", name)
    name = re.sub(r"\s+", " ", name)
    front_face = re.split(r"\s*//\s*", name)[0]
    return front_face.strip()


# %% [markdown]
# These are notebook configurations, depending on how the notebook run is desired:

# %%
# Deck geometry
MAX_DECK_LEN = 115    # EDH decks are exactly 100 cards, but we allow 15 extra slots for companions/sideboards/wishboards.
MAX_QTY_EMBED = 50    # Caps identical card counts. Bumped to 50 to safely handle "Relentless Rats" or "Dragon's Approach" decks.
BATCH_SIZE = 64       # Number of decks processed per gradient update. 64 is the sweet spot for memory vs. gradient stability.
SEED = 42             # Locks all RNG for reproducible training runs.

# Performance and hardware optimizations
USE_AMP = True             # Automatic Mixed Precision. Reduces VRAM and speeds up Tensor Core workloads.
AMP_DTYPE = "bf16"         # bfloat16 is safer than fp16 for deep attention stacks.
USE_ONECYCLE_LR = True     # Smooth warmup/decay schedule for stable multi-phase optimization.
ONECYCLE_WARMUP_PCT = 0.10 # First 10% of steps warm up LR, then cosine decay.

# Define exactly what the notebook should execute when clicking "Run All"
RUN_EMBEDDINGS = False     # True = regenerate fused card embeddings; False = preload existing embedding checkpoint.
RUN_HPARAM_SEARCH = True   # True = run search workflow (if implemented).
RUN_CURRICULUM = True      # True = run full 3-phase curriculum with EDH-first objectives.

MASTER_RUN_NAME = "set_transformer_master_run" # Filename for the final saved PyTorch weights.

DEFAULT_CONFIG = {
    # Architecture Dimensions
    "model_dim": 512,              # Working token dimension after projecting card/qty/role features.
    "hidden_dim": 1024,            # Transformer feedforward width.
    "qty_embed_dim": 32,           # Quantity is useful signal but should remain compact.
    "role_embed_dim": 8,           # Commander/main-deck/pad role indicator dimension.
    "num_heads": 8,                # Multihead attention split across deck-synergy subspaces.
    "num_blocks": 4,               # Depth of contextual set encoding.
    "num_pma_seeds": 4,            # PMA seeds summarize multiple deck facets before scoring.
    "max_qty_embed": MAX_QTY_EMBED,

    # Training Dynamics
    "batch_size": BATCH_SIZE,
    "lr_phase1": 2e-4,             # EDH multitask pretraining LR for deck-structure learning.
    "lr_phase2": 1e-4,             # Supervised regression LR for stable bracket anchoring.
    "lr_phase3": 5e-5,             # Tournament ranking LR for gentle top-end calibration.
    "dropout": 0.10,               # Regularization against package memorization.

    # Loss Function Tuning
    "delta": 0.75,                 # Huber transition point for noisy bracket labels.
    "margin": 0.10,                # Pairwise margin for tournament ordering separation.
    "trainable_last_blocks": 1,    # In Phase 3, unfreeze only last encoder blocks + PMA + output head.
    "cedh_confidence": 1.00,       # Full confidence for tournament placements.
    "gameknights_confidence": 0.35,# Softer confidence: gameplay/politics add outcome noise.

    # Phase 1 multitask mix (MLM + corruption detection + mana infrastructure heads)
    "phase1_w_mlm": 0.50,
    "phase1_w_corruption": 0.30,
    "phase1_w_land_bin": 0.10,
    "phase1_w_ramp_bin": 0.10,
    "phase1_corrupt_swap_ratio": 0.22,
    "phase1_corrupt_land_drop_ratio": 0.35,
    "phase1_land_bin_edges": [30, 34, 37, 40],
    "phase1_ramp_bin_edges": [6, 10, 14, 18],
}

# We use itertools.product to test every combination of these parameters.
grid_space = {
    "margin": [0.10, 0.05, 0.01],
    "delta": [1.00, 0.75, 0.50],
    "dropout": [0.10, 0.2, 0.3],
    "lr_phase3": [5e-5, 1e-5],
}

# Grid Search Data Budgets
SEARCH_SUP_MAX_ITEMS = 45000
SEARCH_TOUR_MAX_ITEMS = 100
SEARCH_PHASE1_EPOCHS = 2
SEARCH_PHASE2_EPOCHS = 3
SEARCH_PHASE3_EPOCHS = 1

# Full-training budgets
TRAIN_UNSUP_MAX_ITEMS = None       # Optional embedding-corpus budget (not used by EDH MLM Phase 1).
TRAIN_SUP_MAX_ITEMS = None         # Phase 1 + Phase 2 source corpus (EDH supervised JSONL).
TRAIN_TOUR_MAX_ITEMS = None        # Tournament ranking corpus budget.
TRAIN_PHASE1_EPOCHS = 8            # Learn EDH deck-language and structural functionality.
TRAIN_PHASE2_EPOCHS = 12           # Anchor continuous 1-5 bracket predictions.
TRAIN_PHASE3_EPOCHS = 5            # Top-end competitive calibration.

# %%
# Initialize hardware device globally so all subsequent cells can use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Global Device set to: {device}")

# %%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# ## 1. Embeddings

# %% [markdown]
# Card vectors are fused from two complementary signals: deck co-occurrence structure (Word2Vec) and card text semantics (MiniLM). These static vectors are then contextualized by the Set Transformer so card meaning can shift based on deck composition and role context.

# %%
if not RUN_EMBEDDINGS: #load embeddings
    blob = torch.load(HYBRID_EMBED_PATH, map_location="cpu")
    vocab = dict(blob["vocab"])
    weights = blob["weights"].float()

    # Backward-compatible patch for older checkpoints that do not include a dedicated <MASK> token.
    if MASK_TOKEN not in vocab:
        vocab[MASK_TOKEN] = weights.shape[0]
        mask_row = torch.zeros(1, weights.shape[1], dtype=weights.dtype)
        weights = torch.cat([weights, mask_row], dim=0)

    VOCAB_SIZE = len(vocab)
    CARD_EMBED_DIM = weights.shape[1]
    PAD_TOKEN_ID = vocab[PAD_TOKEN]
    UNK_TOKEN_ID = vocab[UNK_TOKEN]
    MASK_TOKEN_ID = vocab[MASK_TOKEN]

# %%
if RUN_EMBEDDINGS: #train embeddings
    import gc
    import math
    import logging
    from datasets import Dataset
    from gensim.models import Word2Vec
    from transformers import (
        AutoModelForMaskedLM, AutoTokenizer, AutoModel,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments
    )

    BASE_SEMANTIC_MODEL = "microsoft/MiniLM-L12-H384-uncased"
    MTGJSON_FILE = Path("../../data/oracle_cards.json")
    SEMANTIC_MODEL_DIR = Path("../embeddings/embedding-models/mtg-minilm-mlm")
    W2V_OUTPUT = Path("../embeddings/embedding-models/w2v_mtg_cooccurrence.model")    
    
    W2V_DIM = 512
    W2V_WEIGHT = 0.4
    NLP_WEIGHT = 0.6
    
    def get_combined_oracle_text(card: Dict) -> str:
        """Combines physical stats and Oracle text into a semantic paragraph."""
        chunks = []
        name, cost, t_line = card.get("name", ""), card.get("mana_cost", ""), card.get("type_line", "")
        pt = f"{card.get('power', '')}/{card.get('toughness', '')}".strip("/")
        loyalty = card.get("loyalty", "")

        if name: chunks.append(f"Name: {name}.")
        if cost: chunks.append(f"Mana cost: {cost}.")
        if t_line: chunks.append(f"Type: {t_line}.")
        if pt: chunks.append(f"Stats: {pt}.")
        if loyalty: chunks.append(f"Loyalty: {loyalty}.")

        for face in card.get("card_faces", []):
            f_name, f_cost, f_type = face.get("name", ""), face.get("mana_cost", ""), face.get("type_line", "")
            f_pt = f"{face.get('power', '')}/{face.get('toughness', '')}".strip("/")
            f_summary = " | ".join(p for p in [f_name, f_cost, f_type] if p)
            if f_pt: f_summary += f" | Stats: {f_pt}"
            if f_summary: chunks.append(f"Face: {f_summary}.")

        text_parts = [(" ".join(chunks).strip())]
        if card.get("oracle_text"): text_parts.append(card.get("oracle_text").strip())
        for face in card.get("card_faces", []):
            if face.get("oracle_text"): text_parts.append(face.get("oracle_text").strip())

        # Deduplicate identical faces
        return "\n".join(list(dict.fromkeys(text_parts)))

    print("Parsing Oracle dictionary...")
    with open(MTGJSON_FILE, 'r', encoding='utf-8') as f:
        cards_data = json.load(f)

    oracle_index: Dict[str, str] = {}
    for card in cards_data:
        names = [card.get("name", "")] + [face.get("name", "") for face in card.get("card_faces", [])]
        for name in names:
            if not name: continue
            norm_name = normalize_card_name(name)
            combined_text = get_combined_oracle_text(card)
            
            # Save the richest text version available
            if combined_text.strip() and not oracle_index.get(norm_name, "").strip():
                oracle_index[norm_name] = combined_text


    if not SEMANTIC_MODEL_DIR.exists():
        print(f"\n--- Fine-tuning {BASE_SEMANTIC_MODEL} ---")
        texts = [text for text in oracle_index.values() if text.strip()]
        tokenizer = AutoTokenizer.from_pretrained(BASE_SEMANTIC_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(BASE_SEMANTIC_MODEL)

        tokenized = Dataset.from_dict({"text": texts}).map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=128),
            batched=True, remove_columns=["text"]
        )

        def group_texts(examples):
            concat = {k: sum(examples[k], []) for k in examples.keys()}
            total_len = (len(concat["input_ids"]) // 128) * 128
            return {k: [t[i: i + 128] for i in range(0, total_len, 128)] for k, t in concat.items()}

        lm_dataset = tokenized.map(group_texts, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir=str(SEMANTIC_MODEL_DIR),
            num_train_epochs=2.0,
            per_device_train_batch_size=32,
            learning_rate=5e-5,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=lm_dataset, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(SEMANTIC_MODEL_DIR))
        tokenizer.save_pretrained(str(SEMANTIC_MODEL_DIR))
        
        # Free up VRAM
        del model, trainer, lm_dataset
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n✅ Semantic model already fine-tuned at {SEMANTIC_MODEL_DIR}")

    if not W2V_OUTPUT.exists():
        print("\n--- Loading Corpus into RAM ---")
        # Exploit large RAM by pre-parsing JSON into tokenized deck sentences once.
        deck_sentences = []
        with open(UNSUPERVISED_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # Support both dict and list card containers across scraped corpus variants.
                    row = json.loads(line)
                    cards = row.get("cards", {})
                    if isinstance(cards, dict):
                        deck = list(cards.keys())
                    elif isinstance(cards, list):
                        deck = []
                        for item in cards:
                            if isinstance(item, str):
                                deck.append(item)
                            elif isinstance(item, dict):
                                name = item.get("name")
                                if name:
                                    deck.append(name)
                    else:
                        deck = []
                    if deck:
                        deck_sentences.append(deck)

        print(f"Loaded {len(deck_sentences)} decks into memory.")
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
        print("--- Training Gensim Word2Vec Model ---")

        W2V_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        w2v_model = Word2Vec(
            sentences=deck_sentences,
            vector_size=W2V_DIM,
            window=115,
            min_count=3,
            sg=1,
            workers=12,
            epochs=10,
        )
        w2v_model.save(str(W2V_OUTPUT))

        # Free RAM after training step.
        del deck_sentences
        gc.collect()
    else:
        print(f"✅ Word2Vec model already trained at {W2V_OUTPUT}")
        w2v_model = Word2Vec.load(str(W2V_OUTPUT))

    print("\n--- Fusing Embeddings ---")
    tokenizer = AutoTokenizer.from_pretrained(str(SEMANTIC_MODEL_DIR))
    nlp_model = AutoModel.from_pretrained(str(SEMANTIC_MODEL_DIR)).to(device).eval()
    nlp_dim = nlp_model.config.hidden_size

    vocab = w2v_model.wv.index_to_key
    filtered_vocab = [c for c in vocab if oracle_index.get(normalize_card_name(c), "").strip()]
    print(f"Dropping {len(vocab) - len(filtered_vocab)} cards missing Oracle text.")

    vocab_size = len(filtered_vocab) + 3 # +3 for <PAD>, <UNK>, and <MASK>
    hybrid_dim = W2V_DIM + nlp_dim
    hybrid_matrix = np.zeros((vocab_size, hybrid_dim))
    final_vocab_dict = {PAD_TOKEN: 0, UNK_TOKEN: 1, MASK_TOKEN: 2}

    print("Generating NLP embeddings and concatenating...")
    # 
    for i, card_name in enumerate(filtered_vocab):
        pytorch_id = i + 3
        final_vocab_dict[card_name] = pytorch_id
        
        vec_w2v = w2v_model.wv[card_name]
        n_w2v = np.linalg.norm(vec_w2v)
        if n_w2v > 0: vec_w2v = (vec_w2v / n_w2v) * math.sqrt(W2V_WEIGHT)

        text = oracle_index.get(normalize_card_name(card_name), "")
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = nlp_model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            vec_nlp = (torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)).squeeze(0).cpu().numpy()
            
        n_nlp = np.linalg.norm(vec_nlp)
        if n_nlp > 0: vec_nlp = (vec_nlp / n_nlp) * math.sqrt(NLP_WEIGHT)

        fused = np.concatenate([vec_w2v, vec_nlp])
        n_fused = np.linalg.norm(fused)
        if n_fused > 0: fused = fused / n_fused
            
        hybrid_matrix[pytorch_id] = fused

    # Save to disk for the Set Transformer to load
    torch.save({'weights': torch.FloatTensor(hybrid_matrix), 'vocab': final_vocab_dict}, HYBRID_EMBED_PATH)
    PAD_TOKEN_ID = final_vocab_dict[PAD_TOKEN]
    UNK_TOKEN_ID = final_vocab_dict[UNK_TOKEN]
    MASK_TOKEN_ID = final_vocab_dict[MASK_TOKEN]
    print(f"\n🎉 Saved Hybrid Tensor ({vocab_size}x{hybrid_dim}) to {HYBRID_EMBED_PATH}")
    
    del nlp_model, w2v_model, hybrid_matrix
    gc.collect()
    torch.cuda.empty_cache()

# %%
# Ensure core globals are populated when RUN_EMBEDDINGS=True (freshly generated embeddings path).
if RUN_EMBEDDINGS:
    blob = torch.load(HYBRID_EMBED_PATH, map_location="cpu")
    vocab = dict(blob["vocab"])
    weights = blob["weights"].float()
    VOCAB_SIZE = len(vocab)
    CARD_EMBED_DIM = int(weights.shape[1])
    PAD_TOKEN_ID = vocab[PAD_TOKEN]
    UNK_TOKEN_ID = vocab[UNK_TOKEN]
    MASK_TOKEN_ID = vocab[MASK_TOKEN]
    print(f"Loaded generated embeddings: vocab={VOCAB_SIZE}, dim={CARD_EMBED_DIM}")

# %%
# Oracle-text MLM evaluation was moved out of this training notebook.
# Run it from card_embedding_analysis.ipynb to keep this file focused on EDH grading.
oracle_language_eval = None
print("Oracle-language MLM eval moved to card_embedding_analysis.ipynb")

# %% [markdown]
# ## 2. Data Preprocessing

# %% [markdown]
# We load decklists as unordered sets of card IDs with quantity IDs and a padding mask so the model can attend over variable-size sets. 
# 
# 

# %%
LOADER_NUM_WORKERS = 4
LOADER_PIN_MEMORY = True
LOADER_PERSISTENT_WORKERS = LOADER_NUM_WORKERS > 0
LOADER_PREFETCH_FACTOR = 2

# %%
def _parse_name_qty_items(items: Any) -> List[Tuple[str, int]]:
    parsed: List[Tuple[str, int]] = []
    if isinstance(items, dict):
        for name, qty in items.items():
            if name:
                parsed.append((str(name), int(qty)))
    elif isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                parsed.append((item, 1))
            elif isinstance(item, dict) and item.get("name"):
                parsed.append((str(item["name"]), int(item.get("qty", 1))))
    return parsed


def extract_card_qty_role_triplets(deck_obj: Dict, card_metadata: Optional[Dict] = None) -> List[Tuple[str, int, int]]:
    triplets: List[Tuple[str, int, int]] = []

    # Archidekt-style split zones.
    for cmd_name, cmd_qty in _parse_name_qty_items(deck_obj.get("cmds", [])):
        triplets.append((cmd_name, cmd_qty, 1))

    for zone in ("main", "mainboard", "sideboard"):
        for name, qty in _parse_name_qty_items(deck_obj.get(zone, [])):
            triplets.append((name, qty, 0))

    # Generic cards container.
    for name, qty in _parse_name_qty_items(deck_obj.get("cards", {})):
        triplets.append((name, qty, 0))

    # Collapse duplicates after normalization.
    collapsed: Dict[Tuple[str, int], int] = {}
    for card_name, qty, role_id in triplets:
        normalized = normalize_card_name(card_name)
        qty_int = int(qty)
        if not normalized or qty_int <= 0:
            continue
        key = (normalized, int(role_id))
        collapsed[key] = collapsed.get(key, 0) + qty_int

    return [(name, qty, role_id) for (name, role_id), qty in collapsed.items()]


def extract_commander_names(deck_obj: Dict[str, Any]) -> Tuple[str, ...]:
    commander_candidates: List[str] = []

    for key in ("cmds", "commanders", "commander", "partners"):
        value = deck_obj.get(key)
        if isinstance(value, str):
            commander_candidates.append(value)
        else:
            for name, _ in _parse_name_qty_items(value):
                commander_candidates.append(name)

    normalized = sorted({normalize_card_name(name) for name in commander_candidates if str(name).strip()})
    if normalized:
        return tuple(normalized)
    return ("unknown_commander",)

# %%
class MTGDeckDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, jsonl_path, vocab, max_len=MAX_DECK_LEN, max_items=None, min_cards: int = 10):
        self.vocab = vocab
        self.max_len = max_len
        self.min_cards = int(min_cards)
        self.records: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.commander_keys: List[Tuple[str, ...]] = []

        rows_read = 0
        rows_dropped = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows_read += 1
                data = json.loads(line)

                triplets = extract_card_qty_role_triplets(data)
                if len(triplets) < self.min_cards:
                    rows_dropped += 1
                    continue

                label = float(data.get("bracket", 3.0))
                is_auto = bool(data.get("is_autobracket", False))

                self.samples.append(self._encode_triplets(triplets, label, is_auto))
                self.records.append(data)
                self.commander_keys.append(extract_commander_names(data))

                if max_items and len(self.samples) >= max_items:
                    break

        if not self.samples:
            raise ValueError(f"No valid decks were loaded from {jsonl_path}.")

        print(
            f"Loaded {len(self.samples)} valid supervised decks from {jsonl_path} "
            f"(rows read: {rows_read}, dropped: {rows_dropped})."
        )

    def __len__(self):
        return len(self.samples)

    def _encode_triplets(self, triplets: List[Tuple[str, int, int]], label: float, is_auto: bool) -> Dict[str, torch.Tensor]:
        card_ids: List[int] = []
        qty_ids: List[int] = []
        role_ids: List[int] = []

        for card_name, qty, role in triplets:
            card_ids.append(self.vocab.get(card_name, self.vocab.get(UNK_TOKEN, 1)))
            qty_ids.append(max(1, min(int(qty), MAX_QTY_EMBED)))
            role_ids.append(int(role))

        card_ids = card_ids[: self.max_len]
        qty_ids = qty_ids[: self.max_len]
        role_ids = role_ids[: self.max_len]
        mask = [1] * len(card_ids)

        pad_len = self.max_len - len(card_ids)
        if pad_len > 0:
            card_ids += [PAD_TOKEN_ID] * pad_len
            qty_ids += [0] * pad_len
            role_ids += [2] * pad_len
            mask += [0] * pad_len

        return {
            "card_ids": torch.tensor(card_ids, dtype=torch.long),
            "qty_ids": torch.tensor(qty_ids, dtype=torch.long),
            "role_ids": torch.tensor(role_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": torch.tensor(label, dtype=torch.float32),
            "is_auto": torch.tensor(is_auto, dtype=torch.bool),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

# %%
class TournamentEventDataset(Dataset[List[Dict[str, Any]]]):
    """
    Groups decks by event_url so each item is one full tournament-like event.
    Supports confidence weighting and stores compact raw deck objects for diagnostics.
    """

    def __init__(self, jsonl_path: Path, vocab: Dict[str, int], max_len: int = MAX_DECK_LEN):
        self.vocab = vocab
        self.max_len = max_len
        self.events: Dict[str, List[Dict[str, Any]]] = {}

        rows_read = 0
        rows_dropped = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows_read += 1
                data = json.loads(line)

                event_url = data.get("event_url")
                placement = data.get("placement")
                if not event_url or placement is None:
                    rows_dropped += 1
                    continue

                event_tier = str(data.get("event_tier", "cedh")).lower()
                default_confidence = DEFAULT_CONFIG["gameknights_confidence"] if event_tier == "gameknights" else DEFAULT_CONFIG["cedh_confidence"]

                try:
                    placement_value = float(placement)
                    confidence_value = float(data.get("confidence_weight", default_confidence))
                except (TypeError, ValueError):
                    rows_dropped += 1
                    continue

                triplets = extract_card_qty_role_triplets(data)
                if len(triplets) < 10:
                    rows_dropped += 1
                    continue

                cards_dict: Dict[str, int] = {}
                card_ids: List[int] = []
                qty_ids: List[int] = []
                role_ids: List[int] = []

                for card_name, qty, role in triplets:
                    cards_dict[card_name] = cards_dict.get(card_name, 0) + int(qty)
                    card_ids.append(self.vocab.get(card_name, self.vocab.get(UNK_TOKEN, 1)))
                    qty_ids.append(max(1, min(int(qty), MAX_QTY_EMBED)))
                    role_ids.append(int(role))

                card_ids = card_ids[: self.max_len]
                qty_ids = qty_ids[: self.max_len]
                role_ids = role_ids[: self.max_len]
                mask = [1] * len(card_ids)

                pad_len = self.max_len - len(card_ids)
                if pad_len > 0:
                    card_ids += [PAD_TOKEN_ID] * pad_len
                    qty_ids += [0] * pad_len
                    role_ids += [2] * pad_len
                    mask += [0] * pad_len

                self.events.setdefault(str(event_url), []).append(
                    {
                        "card_ids": card_ids,
                        "qty_ids": qty_ids,
                        "role_ids": role_ids,
                        "mask": mask,
                        "placement": placement_value,
                        "confidence": confidence_value,
                        "event_tier": event_tier,
                        "event_url": str(event_url),
                        "deck_obj": {"cards": cards_dict},
                        "raw_record": data,
                    }
                )

        self.event_list = [decks for decks in self.events.values() if len(decks) >= 2]
        print(
            f"Loaded {len(self.event_list)} valid tournament events from {jsonl_path} "
            f"(rows read: {rows_read}, dropped rows: {rows_dropped})."
        )

    def __len__(self) -> int:
        return len(self.event_list)

    def __getitem__(self, idx: int):
        return self.event_list[idx]


def tournament_collate_fn(batch):
    event_decks = batch[0]

    return {
        "card_ids": torch.tensor([d["card_ids"] for d in event_decks], dtype=torch.long),
        "qty_ids": torch.tensor([d["qty_ids"] for d in event_decks], dtype=torch.long),
        "role_ids": torch.tensor([d["role_ids"] for d in event_decks], dtype=torch.long),
        "mask": torch.tensor([d["mask"] for d in event_decks], dtype=torch.bool),
        "placement": torch.tensor([d["placement"] for d in event_decks], dtype=torch.float32),
        "confidence": torch.tensor([d["confidence"] for d in event_decks], dtype=torch.float32),
        "event_tier": [d["event_tier"] for d in event_decks],
        "event_url": [d["event_url"] for d in event_decks],
    }

# %%
def subset_dataset(dataset: Dataset, max_items: Optional[int] = None, seed: int = SEED):
    if max_items is None or len(dataset) <= max_items:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def _loader_kwargs(shuffle: bool, collate_fn=None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "shuffle": shuffle,
        "num_workers": LOADER_NUM_WORKERS,
        "pin_memory": LOADER_PIN_MEMORY,
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if LOADER_NUM_WORKERS > 0:
        kwargs["persistent_workers"] = LOADER_PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = LOADER_PREFETCH_FACTOR
    return kwargs


def _split_counts(total: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total < 3:
        raise ValueError("Need at least 3 items to build train/val/test splits.")

    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))

    test_count = min(max(1, test_count), total - 2)
    val_count = min(max(1, val_count), total - test_count - 1)
    train_count = total - val_count - test_count
    return train_count, val_count, test_count


def split_supervised_by_commander(
    dataset: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        source_indices = [int(i) for i in dataset.indices]
    else:
        base_dataset = dataset
        source_indices = list(range(len(dataset)))

    if len(source_indices) < 3:
        raise ValueError("Supervised dataset is too small to create train/val/test splits.")
    if not hasattr(base_dataset, "commander_keys"):
        raise AttributeError("Dataset must expose commander_keys for commander-group split.")

    group_to_indices: Dict[Tuple[str, ...], List[int]] = {}
    for idx in source_indices:
        key = tuple(base_dataset.commander_keys[int(idx)])
        group_to_indices.setdefault(key, []).append(int(idx))

    group_keys = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    if len(group_keys) >= 3:
        train_g, val_g, test_g = _split_counts(len(group_keys), val_ratio=val_ratio, test_ratio=test_ratio)
        train_keys = group_keys[:train_g]
        val_keys = group_keys[train_g : train_g + val_g]
        test_keys = group_keys[train_g + val_g :]

        train_indices = [idx for key in train_keys for idx in group_to_indices[key]]
        val_indices = [idx for key in val_keys for idx in group_to_indices[key]]
        test_indices = [idx for key in test_keys for idx in group_to_indices[key]]

        print(
            f"Commander-group split | groups train/val/test = {len(train_keys)}/{len(val_keys)}/{len(test_keys)} "
            f"| decks train/val/test = {len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
        )
    else:
        shuffled_indices = list(source_indices)
        rng.shuffle(shuffled_indices)
        train_n, val_n, test_n = _split_counts(len(shuffled_indices), val_ratio=val_ratio, test_ratio=test_ratio)
        train_indices = shuffled_indices[:train_n]
        val_indices = shuffled_indices[train_n : train_n + val_n]
        test_indices = shuffled_indices[train_n + val_n :]
        print(
            f"Fallback index split (insufficient commander groups) | decks train/val/test = "
            f"{len(train_indices)}/{len(val_indices)}/{len(test_indices)}"
        )

    train_ds = Subset(base_dataset, train_indices)
    val_ds = Subset(base_dataset, val_indices)
    test_ds = Subset(base_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, **_loader_kwargs(shuffle=True))
    val_loader = DataLoader(val_ds, batch_size=batch_size, **_loader_kwargs(shuffle=False))
    test_loader = DataLoader(test_ds, batch_size=batch_size, **_loader_kwargs(shuffle=False))
    return train_loader, val_loader, test_loader


def split_supervised_loader(
    dataset: Dataset,
    val_ratio: float = 0.1,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    train_loader, val_loader, _ = split_supervised_by_commander(
        dataset,
        val_ratio=val_ratio,
        test_ratio=0.1,
        batch_size=batch_size,
        seed=seed,
    )
    return train_loader, val_loader


def split_tournament_loaders(
    dataset: Dataset,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if len(dataset) < 3:
        raise ValueError("Tournament dataset is too small to create train/val/test splits.")

    train_size, val_size, test_size = _split_counts(len(dataset), val_ratio=val_ratio, test_ratio=test_ratio)
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        **_loader_kwargs(shuffle=True, collate_fn=tournament_collate_fn),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        **_loader_kwargs(shuffle=False, collate_fn=tournament_collate_fn),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        **_loader_kwargs(shuffle=False, collate_fn=tournament_collate_fn),
    )
    return train_loader, val_loader, test_loader

# %%
def combine_event_jsonl(input_paths: List[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pairs = set()
    written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for path in input_paths:
            if not path.exists():
                print(f"[WARN] Missing input file: {path}")
                continue

            source_tier = "gameknights" if "gameknights" in path.stem.lower() else "cedh"

            with path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue

                    row = json.loads(line)
                    row.setdefault("event_tier", source_tier)
                    key = (str(row.get("event_url")), str(row.get("deck_url")))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Combined event rows written: {written} -> {output_path}")


combine_event_jsonl(
    [TOURNAMENT_JSONL, GAMEKNIGHTS_JSONL],
    COMBINED_TOURNAMENT_JSONL,
)

print("Loading Supervised EDH Corpus (Phase 1 + Phase 2 source)...")
sup_dataset = MTGDeckDataset(
    jsonl_path=SUPERVISED_JSONL,
    vocab=vocab,
    max_len=MAX_DECK_LEN,
    max_items=TRAIN_SUP_MAX_ITEMS,
)

print("Loading Tournament + GameKnights Corpus (Phase 3 source)...")
tour_dataset = TournamentEventDataset(
    jsonl_path=COMBINED_TOURNAMENT_JSONL,
    vocab=vocab,
    max_len=MAX_DECK_LEN,
)

print("\nCreating DataLoaders...")
sup_train_loader, sup_val_loader, sup_test_loader = split_supervised_by_commander(
    sup_dataset,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

# Phase 1 MLM should only see training decks to avoid leakage from val/test splits.
sup_mlm_loader = DataLoader(
    sup_train_loader.dataset,
    batch_size=BATCH_SIZE,
    **_loader_kwargs(shuffle=True),
)

tour_train_loader, tour_val_loader, tour_test_loader = split_tournament_loaders(
    tour_dataset,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=SEED,
)
tour_eval_loader = tour_val_loader

print("-" * 40)
print(f"EDH MLM Batches (Phase 1 train split): {len(sup_mlm_loader)}")
print(
    f"Supervised Train/Val/Test Batches (Phase 2): "
    f"{len(sup_train_loader)} / {len(sup_val_loader)} / {len(sup_test_loader)}"
)
print(
    f"Tournament+GK Train/Val/Test Events (Phase 3): "
    f"{len(tour_train_loader)} / {len(tour_val_loader)} / {len(tour_test_loader)}"
)
print("-" * 40)

# %% [markdown]
# ## 3. Model Architecture - Set Transformer

# %% [markdown]
# The architecture here uses multiheaded attention layers

# %%
class SetTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        card_embed_dim: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        num_pma_seeds: int,
        max_qty_embed: int,
        qty_embed_dim: int = 32,
        role_embed_dim: int = 8,
        dropout: float = 0.1,
        pretrained_weights: Optional[torch.Tensor] = None,
        land_bin_classes: int = 5,
        ramp_bin_classes: int = 5,
    ):
        super().__init__()

        # 1. Embeddings
        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=PAD_TOKEN_ID)
        if pretrained_weights is not None:
            self.card_embedding.weight.data.copy_(pretrained_weights)

        self.qty_embedding = nn.Embedding(max_qty_embed + 1, qty_embed_dim, padding_idx=0)
        self.role_embedding = nn.Embedding(3, role_embed_dim, padding_idx=2) # 0=Main, 1=Cmd, 2=Pad

        self.input_proj = nn.Sequential(
            nn.Linear(card_embed_dim + qty_embed_dim + role_embed_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        try:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_blocks,
                enable_nested_tensor=False,
            )
        except TypeError:
            # Backward compatibility for older PyTorch versions that do not expose enable_nested_tensor.
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # 3. Pooling by Multihead Attention (PMA)
        self.pma_seeds = nn.Parameter(torch.randn(num_pma_seeds, model_dim))
        self.pma_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 4. Output Heads
        self.dropout = nn.Dropout(dropout)
        pooled_dim = model_dim * num_pma_seeds
        self.output_layer = nn.Linear(pooled_dim, 1)
        self.mlm_head = nn.Linear(model_dim, vocab_size)

        # Phase-1 auxiliary heads (multitask pretraining)
        self.corruption_head = nn.Linear(pooled_dim, 1)
        self.land_bin_head = nn.Linear(pooled_dim, int(land_bin_classes))
        self.ramp_bin_head = nn.Linear(pooled_dim, int(ramp_bin_classes))

    def embed_tokens(self, card_ids: torch.Tensor, qty_ids: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        card_embeds = self.card_embedding(card_ids)
        qty_embeds = self.qty_embedding(qty_ids)
        role_embeds = self.role_embedding(role_ids)
        x = torch.cat([card_embeds, qty_embeds, role_embeds], dim=-1)
        return self.input_proj(x)

    def encode_tokens(self, card_ids: torch.Tensor, qty_ids: torch.Tensor, role_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(card_ids, qty_ids, role_ids)
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=~mask)
        return transformer_out

    def pool(self, token_reps: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        batch_size = token_reps.size(0)
        pma_seeds_expanded = self.pma_seeds.unsqueeze(0).expand(batch_size, -1, -1)

        pma_out, attn = self.pma_attention(
            query=pma_seeds_expanded,
            key=token_reps,
            value=token_reps,
            key_padding_mask=~mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        return pma_out, attn

    def encode_deck_features(self, card_ids: torch.Tensor, qty_ids: torch.Tensor, role_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_reps = self.encode_tokens(card_ids, qty_ids, role_ids, mask)
        pma_out, _ = self.pool(token_reps, mask, return_attention=False)
        return self.dropout(pma_out.reshape(card_ids.size(0), -1))

    def forward_mlm(self, card_ids: torch.Tensor, qty_ids: torch.Tensor, role_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformer_out = self.encode_tokens(card_ids, qty_ids, role_ids, mask)
        return self.mlm_head(transformer_out)

    def forward_phase1_aux(self, card_ids: torch.Tensor, qty_ids: torch.Tensor, role_ids: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode_deck_features(card_ids, qty_ids, role_ids, mask)
        return {
            "corruption": self.corruption_head(features).squeeze(-1),
            "land_bin": self.land_bin_head(features),
            "ramp_bin": self.ramp_bin_head(features),
        }

    def forward_raw(
        self,
        card_ids: torch.Tensor,
        qty_ids: torch.Tensor,
        role_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        deck_features = self.encode_deck_features(card_ids, qty_ids, role_ids, mask)
        return self.output_layer(deck_features).squeeze(-1)

    def forward(
        self,
        card_ids: torch.Tensor,
        qty_ids: torch.Tensor,
        role_ids: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ):
        transformer_out = self.encode_tokens(card_ids, qty_ids, role_ids, mask)
        pma_out, attn = self.pool(transformer_out, mask, return_attention=return_attention)

        pma_out_flat = self.dropout(pma_out.reshape(card_ids.size(0), -1))
        raw = self.output_layer(pma_out_flat).squeeze(-1)
        scores = 1.0 + 4.0 * torch.sigmoid(raw)

        if return_attention:
            return scores, attn
        return scores

# %%
model = SetTransformer(
    vocab_size=VOCAB_SIZE,
    card_embed_dim=CARD_EMBED_DIM,
    model_dim=DEFAULT_CONFIG["model_dim"],
    hidden_dim=DEFAULT_CONFIG["hidden_dim"],
    num_heads=DEFAULT_CONFIG["num_heads"],
    num_blocks=DEFAULT_CONFIG["num_blocks"],
    num_pma_seeds=DEFAULT_CONFIG["num_pma_seeds"],
    max_qty_embed=DEFAULT_CONFIG["max_qty_embed"],
    qty_embed_dim=DEFAULT_CONFIG["qty_embed_dim"],
    role_embed_dim=DEFAULT_CONFIG["role_embed_dim"],
    dropout=DEFAULT_CONFIG["dropout"],
    pretrained_weights=weights,
    land_bin_classes=len(DEFAULT_CONFIG.get("phase1_land_bin_edges", [30, 34, 37, 40])) + 1,
    ramp_bin_classes=len(DEFAULT_CONFIG.get("phase1_ramp_bin_edges", [6, 10, 14, 18])) + 1,
).to(device)

# %% [markdown]
# ## 4. Training

# %% [markdown]
# Here we run the 3 phases of training, but first, some evaluation metrics and build:

# %%
def build_model(config: Dict[str, Any]) -> SetTransformer:
    return SetTransformer(
        vocab_size=VOCAB_SIZE,
        card_embed_dim=CARD_EMBED_DIM,
        model_dim=int(config.get("model_dim", 512)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        num_heads=int(config.get("num_heads", 8)),
        num_blocks=int(config.get("num_blocks", 4)),
        num_pma_seeds=int(config.get("num_pma_seeds", 4)),
        max_qty_embed=int(config.get("max_qty_embed", MAX_QTY_EMBED)),
        qty_embed_dim=int(config.get("qty_embed_dim", 32)),
        role_embed_dim=int(config.get("role_embed_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        pretrained_weights=weights,
        land_bin_classes=len(config.get("phase1_land_bin_edges", [30, 34, 37, 40])) + 1,
        ramp_bin_classes=len(config.get("phase1_ramp_bin_edges", [6, 10, 14, 18])) + 1,
    ).to(device)


RESULTS_DIR = CHECKPOINT_DIR / "analysis"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def _ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json_artifact(filename: str, payload: Any) -> Path:
    _ensure_results_dirs()
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)
    print(f"Saved JSON artifact: {path}")
    return path


def save_table_artifact(filename: str, rows: List[Dict[str, Any]]) -> Path:
    import csv

    _ensure_results_dirs()
    path = TABLES_DIR / filename
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        print(f"Saved empty table artifact: {path}")
        return path

    fieldnames = sorted({str(k) for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_serializable(row.get(k)) for k in fieldnames})
    print(f"Saved table artifact: {path}")
    return path


def save_figure_artifact(filename: str, fig=None, close: bool = True, dpi: int = 160) -> Path:
    _ensure_results_dirs()
    if fig is None:
        fig = plt.gcf()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"Saved figure artifact: {path}")
    return path


def make_onecycle_scheduler(optimizer: optim.Optimizer, lr: float, epochs: int, steps_per_epoch: int):
    if not USE_ONECYCLE_LR or steps_per_epoch <= 0:
        return None
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=ONECYCLE_WARMUP_PCT,
        anneal_strategy="cos",
    )


def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], save: bool = True):
    if not histories:
        print("No histories to plot.")
        return []

    saved_paths: List[str] = []
    for phase_name, history in histories.items():
        metric_keys = [k for k, v in history.items() if isinstance(v, list) and v]
        if not metric_keys:
            continue

        fig = plt.figure(figsize=(9, 4.5))
        for key in metric_keys:
            plt.plot(history[key], label=key)
        plt.title(phase_name)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save:
            phase_slug = str(phase_name).replace(" ", "_").lower()
            path = save_figure_artifact(f"training_curve_{phase_slug}.png", fig=fig, close=False)
            saved_paths.append(str(path))
        plt.show()
        plt.close(fig)

    if save:
        save_json_artifact("training_histories.json", histories)
    return saved_paths


def _event_tau_for_batch(scores: np.ndarray, actual_placements: np.ndarray) -> float:
    tau, _ = kendalltau(scores, -actual_placements)
    return float(tau) if not np.isnan(tau) else float("nan")


def compute_tournament_tau_stats(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    """
    Evaluates the model against the cEDH tournament dataset using Kendall's Rank Correlation.
    """
    model.eval()
    all_taus = []

    with torch.no_grad():
        for batch in loader:
            card_ids = batch["card_ids"].to(device)
            qty_ids = batch["qty_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            mask = batch["mask"].to(device)
            actual_placements = batch["placement"].cpu().numpy()

            # Kendall Tau requires at least 3 items to be mathematically meaningful
            if len(actual_placements) < 3:
                continue

            # Predict deck power
            predicted_scores = model(card_ids, qty_ids, role_ids, mask).squeeze().cpu().numpy()

            # Negate actual placements because lower placement (1st) is better (higher predicted score)
            tau, _ = kendalltau(predicted_scores, -actual_placements)
            if not np.isnan(tau):
                all_taus.append(float(tau))

    if not all_taus:
        return {"mean_tau": float("nan"), "median_tau": float("nan"), "positive_corr_pct": 0.0}

    tau_array = np.array(all_taus, dtype=np.float32)
    return {
        "mean_tau": float(tau_array.mean()),
        "median_tau": float(np.median(tau_array)),
        "positive_corr_pct": float((tau_array > 0).mean() * 100.0),
    }


# Quick sanity check initialization
test_model = build_model(DEFAULT_CONFIG)
print(f"Model initialized with {sum(p.numel() for p in test_model.parameters() if p.requires_grad):,} trainable parameters.")

# %% [markdown]
# ### Phase 1: Decklist Pre-training (Unsupervised MLM)
# 
# Before the model tries to predict a single power level score, it needs to learn how MTG cards relate to each other. 
# In this phase, we take an EDH deck, randomly mask out 15% of the cards, and force the model to guess what is missing. 
# 
# **Why we do this:**
# By guessing missing cards, the Self-Attention heads learn competitive deckbuilding patterns and combo packages (e.g., if Thassa's Oracle and Demonic Consultation are in the deck, the masked card is likely Force of Will, not Colossal Dreadmaw). This builds a powerful "synergy engine" before we ever introduce the subjective 1-5 power brackets.

# %% [markdown]
# Phase 1 is EDH-only in practice: it uses the supervised EDH corpus loader for MLM so token co-context is Commander-consistent and not diluted by cross-format deck grammar.

# %%
def apply_mlm_mask(card_ids: torch.Tensor, mask: torch.Tensor, mask_token_id: int = MASK_TOKEN_ID, mask_ratio: float = 0.15):
    # Use a dedicated <MASK> token so the model can distinguish masked cards from genuine unknown cards.
    masked_ids = card_ids.clone()
    labels = torch.full_like(card_ids, fill_value=-100) # -100 is ignored by CrossEntropyLoss

    for i in range(card_ids.size(0)):
        valid_positions = torch.nonzero(mask[i], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        n_mask = max(1, int(valid_positions.numel() * mask_ratio))
        perm = torch.randperm(valid_positions.numel(), device=card_ids.device)[:n_mask]
        chosen = valid_positions[perm]

        labels[i, chosen] = card_ids[i, chosen]
        masked_ids[i, chosen] = mask_token_id

    return masked_ids, labels


def _decode_card_id(card_id: int, id_to_card: Dict[int, str]) -> str:
    return id_to_card.get(int(card_id), f"<ID:{int(card_id)}>")


def _build_phase1_feature_lookups(
    vocab: Dict[str, int],
    vocab_size: int,
    oracle_json: Path = Path("../../data/oracle_cards.json"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    land_ids: set[int] = set()
    ramp_ids: set[int] = set()

    ramp_keywords = [
        "add {",
        "search your library for a land",
        "treasure token",
        "create a treasure",
        "create two treasure",
        "mana of any color",
        "costs {1} less",
    ]

    try:
        with oracle_json.open("r", encoding="utf-8") as f:
            cards_data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not load oracle file for phase1 auxiliaries: {exc}")
        return torch.zeros(vocab_size, dtype=torch.bool), torch.zeros(vocab_size, dtype=torch.bool)

    for card in cards_data:
        names = [card.get("name", "")] + [face.get("name", "") for face in card.get("card_faces", [])]
        type_lines = [str(card.get("type_line", ""))] + [str(face.get("type_line", "")) for face in card.get("card_faces", [])]

        text_parts = [str(card.get("oracle_text", ""))]
        text_parts.extend(str(face.get("oracle_text", "")) for face in card.get("card_faces", []))
        text_blob = " ".join(text_parts).lower()

        is_land = any("land" in tl.lower() for tl in type_lines if tl)
        is_ramp = any(keyword in text_blob for keyword in ramp_keywords)

        for name in names:
            if not str(name).strip():
                continue
            norm = normalize_card_name(name)
            cid = vocab.get(norm)
            if cid is None:
                continue
            if is_land:
                land_ids.add(int(cid))
            if is_ramp:
                ramp_ids.add(int(cid))

    land_lookup = torch.zeros(vocab_size, dtype=torch.bool)
    ramp_lookup = torch.zeros(vocab_size, dtype=torch.bool)
    if land_ids:
        land_lookup[list(land_ids)] = True
    if ramp_ids:
        ramp_lookup[list(ramp_ids)] = True

    return land_lookup, ramp_lookup


def _count_feature_cards(
    card_ids: torch.Tensor,
    qty_ids: torch.Tensor,
    mask: torch.Tensor,
    feature_lookup: torch.Tensor,
) -> torch.Tensor:
    local_lookup = feature_lookup.to(card_ids.device)
    feature_mask = local_lookup[card_ids] & mask
    return (qty_ids.float() * feature_mask.float()).sum(dim=1)


def _counts_to_bins(counts: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(counts, bin_edges)


def _corrupt_deck_batch(
    card_ids: torch.Tensor,
    qty_ids: torch.Tensor,
    role_ids: torch.Tensor,
    mask: torch.Tensor,
    land_lookup: torch.Tensor,
    swap_ratio: float,
    land_drop_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    corrupted_card_ids = card_ids.clone()
    corrupted_qty_ids = qty_ids.clone()
    corrupted_role_ids = role_ids.clone()
    corrupted_mask = mask.clone()

    local_land_lookup = land_lookup.to(card_ids.device)
    batch_size = card_ids.size(0)

    for i in range(batch_size):
        valid_positions = torch.nonzero(corrupted_mask[i], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        main_positions = valid_positions[corrupted_role_ids[i, valid_positions] == 0]
        if main_positions.numel() < 2:
            main_positions = valid_positions

        if batch_size > 1:
            donor_idx = random.randrange(batch_size - 1)
            if donor_idx >= i:
                donor_idx += 1
        else:
            donor_idx = i

        donor_valid = torch.nonzero(mask[donor_idx], as_tuple=False).flatten()
        donor_main = donor_valid[role_ids[donor_idx, donor_valid] == 0]
        if donor_main.numel() < 2:
            donor_main = donor_valid

        changed_any = False

        swap_count = int(max(2, round(float(main_positions.numel()) * float(swap_ratio))))
        swap_count = min(swap_count, int(main_positions.numel()), int(donor_main.numel()))
        if swap_count > 0:
            swap_positions = main_positions[torch.randperm(main_positions.numel(), device=card_ids.device)[:swap_count]]
            donor_positions = donor_main[torch.randperm(donor_main.numel(), device=card_ids.device)[:swap_count]]

            corrupted_card_ids[i, swap_positions] = card_ids[donor_idx, donor_positions]
            corrupted_qty_ids[i, swap_positions] = qty_ids[donor_idx, donor_positions]
            # Preserve role layout so corruption remains superficially plausible.
            corrupted_role_ids[i, swap_positions] = role_ids[i, swap_positions]
            changed_any = True

        valid_after_swap = torch.nonzero(corrupted_mask[i], as_tuple=False).flatten()
        land_positions = valid_after_swap[local_land_lookup[corrupted_card_ids[i, valid_after_swap]]]

        if land_positions.numel() > 0:
            drop_count = int(round(float(land_positions.numel()) * float(land_drop_ratio)))
            drop_count = max(1, drop_count)
            drop_count = min(drop_count, int(land_positions.numel()))
            drop_positions = land_positions[torch.randperm(land_positions.numel(), device=card_ids.device)[:drop_count]]

            corrupted_card_ids[i, drop_positions] = PAD_TOKEN_ID
            corrupted_qty_ids[i, drop_positions] = 0
            corrupted_role_ids[i, drop_positions] = 2
            corrupted_mask[i, drop_positions] = False
            changed_any = True

        # Ensure every sample has at least one perturbation.
        if not changed_any and main_positions.numel() > 0 and donor_main.numel() > 0:
            corrupted_card_ids[i, main_positions[0]] = card_ids[donor_idx, donor_main[0]]
            corrupted_qty_ids[i, main_positions[0]] = qty_ids[donor_idx, donor_main[0]]

    return corrupted_card_ids, corrupted_qty_ids, corrupted_role_ids, corrupted_mask


def show_phase1_mlm_example(
    model: nn.Module,
    loader: DataLoader,
    vocab: Dict[str, int],
    mask_ratio: float = 0.15,
    topk: int = 5,
    sample_index: int = 0,
    max_display_cards: int = 60,
) -> Dict[str, Any]:
    """
    Prints a single MLM example showing:
    1) Original card list
    2) Masked card list
    3) Top-k model guesses for each masked position
    """
    was_training = model.training
    model.eval()

    id_to_card = {idx: name for name, idx in vocab.items()}
    batch = next(iter(loader))

    card_ids = batch["card_ids"].to(device)
    qty_ids = batch["qty_ids"].to(device)
    role_ids = batch["role_ids"].to(device)
    mask = batch["mask"].to(device)

    sample_index = int(max(0, min(sample_index, card_ids.size(0) - 1)))
    card_row = card_ids[sample_index:sample_index + 1]
    qty_row = qty_ids[sample_index:sample_index + 1]
    role_row = role_ids[sample_index:sample_index + 1]
    mask_row = mask[sample_index:sample_index + 1]

    masked_ids, labels = apply_mlm_mask(
        card_row,
        mask_row,
        mask_token_id=MASK_TOKEN_ID,
        mask_ratio=mask_ratio,
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16, enabled=USE_AMP):
            logits = model.forward_mlm(masked_ids, qty_row, role_row, mask_row)

    valid_positions = torch.nonzero(mask_row[0], as_tuple=False).flatten().tolist()
    masked_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten().tolist()

    original_cards = [_decode_card_id(card_row[0, pos].item(), id_to_card) for pos in valid_positions]
    masked_cards = [
        MASK_TOKEN if pos in masked_positions else _decode_card_id(card_row[0, pos].item(), id_to_card)
        for pos in valid_positions
    ]

    print("\n=== Phase 1 MLM Example ===")
    print(f"Deck length (unpadded): {len(valid_positions)}")

    shown_n = min(len(valid_positions), max_display_cards)
    print(f"\nOriginal list (first {shown_n} cards):")
    for i, name in enumerate(original_cards[:shown_n], start=1):
        print(f"{i:>2}. {name}")

    print(f"\nMasked list (first {shown_n} cards):")
    for i, name in enumerate(masked_cards[:shown_n], start=1):
        print(f"{i:>2}. {name}")

    print("\nGuesses per masked position:")
    predictions = []
    probs = torch.softmax(logits[0], dim=-1)

    for pos in masked_positions:
        true_id = int(labels[0, pos].item())
        true_card = _decode_card_id(true_id, id_to_card)
        top_vals, top_ids = torch.topk(probs[pos], k=int(topk))
        guessed = [
            (_decode_card_id(int(cid.item()), id_to_card), float(p.item()))
            for cid, p in zip(top_ids, top_vals)
        ]
        predictions.append({
            "position": int(pos),
            "true_card": true_card,
            "guesses": guessed,
        })

        print(f"- Position {pos}: true = {true_card}")
        for rank, (guess_name, guess_prob) in enumerate(guessed, start=1):
            print(f"    {rank}. {guess_name} ({guess_prob:.4f})")

    if was_training:
        model.train()

    return {
        "original_cards": original_cards,
        "masked_cards": masked_cards,
        "predictions": predictions,
    }


def train_phase1_mlm(model: nn.Module, loader: DataLoader, epochs: int, lr: float):
    print("\n--- Starting Phase 1: EDH Multitask Pre-training ---")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = make_onecycle_scheduler(optimizer, lr=lr, epochs=epochs, steps_per_epoch=len(loader))

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    corruption_criterion = nn.BCEWithLogitsLoss()
    aux_criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = {
        "train_loss": [],
        "mlm_loss": [],
        "corruption_loss": [],
        "land_loss": [],
        "ramp_loss": [],
        "lr": [],
    }

    land_lookup, ramp_lookup = _build_phase1_feature_lookups(vocab, VOCAB_SIZE)
    land_lookup = land_lookup.to(device)
    ramp_lookup = ramp_lookup.to(device)
    print(
        f"Phase1 feature lookup coverage | land_ids={int(land_lookup.sum().item())} "
        f"| ramp_ids={int(ramp_lookup.sum().item())}"
    )

    land_edges = torch.tensor(DEFAULT_CONFIG.get("phase1_land_bin_edges", [30, 34, 37, 40]), device=device, dtype=torch.float32)
    ramp_edges = torch.tensor(DEFAULT_CONFIG.get("phase1_ramp_bin_edges", [6, 10, 14, 18]), device=device, dtype=torch.float32)

    w_mlm = float(DEFAULT_CONFIG.get("phase1_w_mlm", 0.50))
    w_corruption = float(DEFAULT_CONFIG.get("phase1_w_corruption", 0.30))
    w_land = float(DEFAULT_CONFIG.get("phase1_w_land_bin", 0.10))
    w_ramp = float(DEFAULT_CONFIG.get("phase1_w_ramp_bin", 0.10))

    corrupt_swap_ratio = float(DEFAULT_CONFIG.get("phase1_corrupt_swap_ratio", 0.22))
    corrupt_land_drop_ratio = float(DEFAULT_CONFIG.get("phase1_corrupt_land_drop_ratio", 0.35))

    model.train()

    for epoch in range(epochs):
        running_total = 0.0
        running_mlm = 0.0
        running_corruption = 0.0
        running_land = 0.0
        running_ramp = 0.0
        valid_steps = 0

        progress_bar = tqdm(loader, desc=f"[Phase 1 Multitask] Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            card_ids = batch["card_ids"].to(device)
            qty_ids = batch["qty_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            mask = batch["mask"].to(device)

            masked_ids, labels = apply_mlm_mask(card_ids, mask)
            if not torch.any(labels != -100):
                continue

            corrupted_card_ids, corrupted_qty_ids, corrupted_role_ids, corrupted_mask = _corrupt_deck_batch(
                card_ids,
                qty_ids,
                role_ids,
                mask,
                land_lookup=land_lookup,
                swap_ratio=corrupt_swap_ratio,
                land_drop_ratio=corrupt_land_drop_ratio,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16, enabled=USE_AMP):
                mlm_logits = model.forward_mlm(masked_ids, qty_ids, role_ids, mask)
                mlm_loss = mlm_criterion(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

                real_aux = model.forward_phase1_aux(card_ids, qty_ids, role_ids, mask)
                corrupt_aux = model.forward_phase1_aux(
                    corrupted_card_ids,
                    corrupted_qty_ids,
                    corrupted_role_ids,
                    corrupted_mask,
                )

                corruption_logits = torch.cat([real_aux["corruption"], corrupt_aux["corruption"]], dim=0)
                corruption_targets = torch.cat(
                    [
                        torch.ones(card_ids.size(0), device=device),
                        torch.zeros(card_ids.size(0), device=device),
                    ],
                    dim=0,
                )
                corruption_loss = corruption_criterion(corruption_logits, corruption_targets)

                land_counts = _count_feature_cards(card_ids, qty_ids, mask, land_lookup)
                ramp_counts = _count_feature_cards(card_ids, qty_ids, mask, ramp_lookup)
                land_targets = _counts_to_bins(land_counts, land_edges).long()
                ramp_targets = _counts_to_bins(ramp_counts, ramp_edges).long()

                land_loss = aux_criterion(real_aux["land_bin"], land_targets)
                ramp_loss = aux_criterion(real_aux["ramp_bin"], ramp_targets)

                total_loss = (
                    w_mlm * mlm_loss
                    + w_corruption * corruption_loss
                    + w_land * land_loss
                    + w_ramp * ramp_loss
                )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            running_total += float(total_loss.item())
            running_mlm += float(mlm_loss.item())
            running_corruption += float(corruption_loss.item())
            running_land += float(land_loss.item())
            running_ramp += float(ramp_loss.item())
            valid_steps += 1

            progress_bar.set_postfix(
                {
                    "total": f"{total_loss.item():.4f}",
                    "mlm": f"{mlm_loss.item():.4f}",
                    "corr": f"{corruption_loss.item():.4f}",
                }
            )

        avg_total = running_total / max(1, valid_steps)
        avg_mlm = running_mlm / max(1, valid_steps)
        avg_corruption = running_corruption / max(1, valid_steps)
        avg_land = running_land / max(1, valid_steps)
        avg_ramp = running_ramp / max(1, valid_steps)

        history["train_loss"].append(avg_total)
        history["mlm_loss"].append(avg_mlm)
        history["corruption_loss"].append(avg_corruption)
        history["land_loss"].append(avg_land)
        history["ramp_loss"].append(avg_ramp)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Phase 1 | Epoch {epoch + 1} | total={avg_total:.4f} | mlm={avg_mlm:.4f} "
            f"| corr={avg_corruption:.4f} | land={avg_land:.4f} | ramp={avg_ramp:.4f} "
            f"| valid_steps={valid_steps}"
        )

    return history

# %% [markdown]
# ### Phase 2: Power Bracket Anchoring (Supervised Huber Regression)
# 
# Now that the model understands MTG synergy, we swap the MLM head for the regression output layer. 
# We train the model on the Moxfield dataset to predict the 1.0 to 5.0 power bracket labels.
# 
# **Why Huber Loss?**
# Crowdsourced power levels are noisy and highly subjective (everyone thinks their deck is a "7/10"). Huber Loss acts like Mean Squared Error (MSE) for small errors, but switches to Mean Absolute Error (MAE) for large errors. This prevents the model from destroying its learned synergy engine just to satisfy a single user who incorrectly labeled their cEDH deck as a 1.0 jank pile.

# %%
def deck_to_model_inputs(deck_obj: Dict[str, Any], max_len: int = MAX_DECK_LEN):
    triplets = extract_card_qty_role_triplets(deck_obj)
    if len(triplets) < 1:
        raise ValueError("Deck has no valid cards after preprocessing.")

    used_names, card_ids, qty_ids, role_ids = [], [], [], []
    for card_name, qty, role in triplets[:max_len]:
        used_names.append(card_name)
        card_ids.append(vocab.get(card_name, UNK_TOKEN_ID))
        qty_ids.append(max(1, min(int(qty), MAX_QTY_EMBED)))
        role_ids.append(role)

    mask = [1] * len(card_ids)
    pad_len = max_len - len(card_ids)
    if pad_len > 0:
        card_ids += [PAD_TOKEN_ID] * pad_len
        qty_ids += [0] * pad_len
        role_ids += [2] * pad_len
        mask += [0] * pad_len

    batch = {
        "card_ids": torch.tensor([card_ids], dtype=torch.long, device=device),
        "qty_ids": torch.tensor([qty_ids], dtype=torch.long, device=device),
        "role_ids": torch.tensor([role_ids], dtype=torch.long, device=device),
        "mask": torch.tensor([mask], dtype=torch.bool, device=device),
    }
    return batch, used_names


def evaluate_regression(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            card_ids = batch["card_ids"].to(device)
            qty_ids = batch["qty_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16, enabled=USE_AMP):
                pred = model(card_ids, qty_ids, role_ids, mask)
                loss = loss_fn(pred, target)
            total += float(loss.item())
    model.train()
    return total / max(1, len(loader))


def train_phase2_huber(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    delta: float,
    best_ckpt_path: Optional[Path] = None,
):
    print("\n--- Starting Phase 2: Supervised Huber Regression ---")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = make_onecycle_scheduler(optimizer, lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    train_loss_fn = nn.HuberLoss(delta=delta, reduction="none")
    eval_loss_fn = nn.HuberLoss(delta=delta, reduction="mean")
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = {"train_loss": [], "val_loss": [], "lr": []}

    best_val_loss = float("inf")
    best_epoch = -1
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"[Phase 2 Huber] Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            card_ids = batch["card_ids"].to(device)
            qty_ids = batch["qty_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            is_auto = batch["is_auto"].to(device)
            weights_local = torch.where(is_auto, 0.5, 1.0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16, enabled=USE_AMP):
                pred = model(card_ids, qty_ids, role_ids, mask)
                unreduced_loss = train_loss_fn(pred, target)
                loss = (unreduced_loss * weights_local).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            train_loss += float(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_avg = train_loss / max(1, len(train_loader))
        val_loss = evaluate_regression(model, val_loader, eval_loss_fn)
        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            if best_ckpt_path is not None:
                best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "phase": "phase2",
                        "epoch": best_epoch,
                        "val_loss": best_val_loss,
                        "model_state_dict": model.state_dict(),
                        "config": dict(DEFAULT_CONFIG),
                        "vocab_size": VOCAB_SIZE,
                        "max_qty": MAX_QTY_EMBED,
                    },
                    best_ckpt_path,
                )

        print(f"Phase 2 | Epoch {epoch + 1} | Train Loss: {train_avg:.4f} | Val Loss: {val_loss:.4f}")

    history["best_val_loss"] = [best_val_loss]
    history["best_epoch"] = [float(best_epoch)]
    if best_ckpt_path is not None:
        print(
            f"Phase 2 best checkpoint | epoch={best_epoch} | val_loss={best_val_loss:.4f} "
            f"| path={best_ckpt_path}"
        )

    return history

# %% [markdown]
# ### Phase 3: cEDH Tournament Calibration (Pairwise Ranking)
# 
# Human labels can only get us so far. To truly understand the peak of competitive MTG, we freeze the base representation layers and only train the final output block. 
# 
# We feed the model decks from actual cEDH tournaments. Instead of predicting a specific number, we use a Pairwise Margin Loss: if Deck A placed higher than Deck B in the tournament, the model's predicted score for Deck A *must* be higher than Deck B by a specific margin. 
# 
# **Why we do this:**
# This natively calibrates the very top end of our 1-5 scale to reflect actual tournament winning percentages, creating objective separation between "High Power Casual" and true "cEDH".

# %%
def configure_phase3_trainability(model: SetTransformer, trainable_last_blocks: int = 1):
    for param in model.parameters():
        param.requires_grad = False

    if trainable_last_blocks > 0:
        for layer in model.transformer_encoder.layers[-trainable_last_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    for param in model.pma_attention.parameters():
        param.requires_grad = True
    model.pma_seeds.requires_grad = True
    for param in model.output_layer.parameters():
        param.requires_grad = True


def train_phase3_tournament(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    lr: float,
    margin: float,
    trainable_last_blocks: int = 1,
    best_ckpt_path: Optional[Path] = None,
):
    print("\n--- Starting Phase 3: Tournament Pairwise Calibration ---")
    configure_phase3_trainability(model, trainable_last_blocks=trainable_last_blocks)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = make_onecycle_scheduler(optimizer, lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = {"train_loss": [], "mean_tau": [], "positive_corr_pct": [], "lr": []}

    best_tau = -float("inf")
    best_epoch = -1
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_events = 0

        progress_bar = tqdm(train_loader, desc=f"[Phase 3 Pairwise] Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            card_ids = batch["card_ids"].to(device)
            qty_ids = batch["qty_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            mask = batch["mask"].to(device)
            placements = batch["placement"].to(device)
            confidences = batch["confidence"].to(device)

            num_decks = card_ids.size(0)
            if num_decks < 2:
                continue

            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16, enabled=USE_AMP):
                if hasattr(model, "forward_raw"):
                    scores = model.forward_raw(card_ids, qty_ids, role_ids, mask)
                else:
                    scores = model(card_ids, qty_ids, role_ids, mask)

                pod_loss = torch.tensor(0.0, device=device)
                pairs_in_pod = 0

                for i in range(num_decks):
                    for j in range(i + 1, num_decks):
                        if placements[i] == placements[j]:
                            continue

                        if placements[i] < placements[j]:
                            win_idx, lose_idx = i, j
                        else:
                            win_idx, lose_idx = j, i

                        pair_loss = torch.relu(margin - (scores[win_idx] - scores[lose_idx]))
                        pair_confidence = (confidences[win_idx] + confidences[lose_idx]) / 2.0
                        pod_loss += pair_loss * pair_confidence
                        pairs_in_pod += 1

                if pairs_in_pod == 0:
                    continue

                pod_loss = pod_loss / pairs_in_pod

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(pod_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += float(pod_loss.item())
            epoch_events += 1
            progress_bar.set_postfix({"loss": f"{epoch_loss / max(1, epoch_events):.4f}"})

        stats = compute_tournament_tau_stats(model, eval_loader)
        mean_loss = epoch_loss / max(1, epoch_events)
        history["train_loss"].append(mean_loss)
        history["mean_tau"].append(stats["mean_tau"])
        history["positive_corr_pct"].append(stats["positive_corr_pct"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if stats["mean_tau"] > best_tau:
            best_tau = float(stats["mean_tau"])
            best_epoch = epoch + 1
            if best_ckpt_path is not None:
                best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "phase": "phase3",
                        "epoch": best_epoch,
                        "mean_tau": best_tau,
                        "model_state_dict": model.state_dict(),
                        "config": dict(DEFAULT_CONFIG),
                        "vocab_size": VOCAB_SIZE,
                        "max_qty": MAX_QTY_EMBED,
                    },
                    best_ckpt_path,
                )

        print(
            f"Phase 3 | Epoch {epoch + 1} | Loss: {mean_loss:.4f} "
            f"| Kendall Tau: {stats['mean_tau']:.3f} | Pos Corr: {stats['positive_corr_pct']:.1f}%"
        )

    history["best_mean_tau"] = [best_tau]
    history["best_epoch"] = [float(best_epoch)]
    if best_ckpt_path is not None:
        print(
            f"Phase 3 best checkpoint | epoch={best_epoch} | mean_tau={best_tau:.4f} "
            f"| path={best_ckpt_path}"
        )

    return history

# %%
def run_hparam_search(
    base_config: Dict[str, Any],
    grid: Dict[str, List[Any]],
    max_trials: Optional[int] = 12,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not grid:
        return dict(base_config), []

    keys = list(grid.keys())
    all_values = [grid[k] for k in keys]
    candidates = list(itertools.product(*all_values))
    if max_trials is not None:
        candidates = candidates[:max_trials]

    print(f"\n--- Hyperparameter Search: {len(candidates)} trial(s) ---")
    results: List[Dict[str, Any]] = []

    # Search loaders are budgeted to keep total runtime manageable.
    search_sup_dataset = subset_dataset(sup_dataset, SEARCH_SUP_MAX_ITEMS, seed=SEED)
    search_tour_dataset = subset_dataset(tour_dataset, SEARCH_TOUR_MAX_ITEMS, seed=SEED)

    search_sup_train_loader, search_sup_val_loader, _ = split_supervised_by_commander(
        search_sup_dataset,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # Keep Phase 1 on the search-train split only to avoid leakage into search val/test.
    search_phase1_loader = DataLoader(
        search_sup_train_loader.dataset,
        batch_size=BATCH_SIZE,
        **_loader_kwargs(shuffle=True),
    )

    search_tour_train_loader, search_tour_eval_loader, _ = split_tournament_loaders(
        search_tour_dataset,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=SEED,
    )

    best_score = -1e9
    best_config = dict(base_config)

    for trial_idx, values in enumerate(candidates, start=1):
        trial_config = dict(base_config)
        for k, v in zip(keys, values):
            trial_config[k] = v

        print(f"\n[Search {trial_idx}/{len(candidates)}] config={{" + ", ".join([f"{k}={trial_config[k]}" for k in keys]) + "}}")
        trial_model = build_model(trial_config)

        _ = train_phase1_mlm(
            trial_model,
            search_phase1_loader,
            epochs=SEARCH_PHASE1_EPOCHS,
            lr=trial_config["lr_phase1"],
        )
        h2 = train_phase2_huber(
            trial_model,
            search_sup_train_loader,
            search_sup_val_loader,
            epochs=SEARCH_PHASE2_EPOCHS,
            lr=trial_config["lr_phase2"],
            delta=trial_config["delta"],
        )
        h3 = train_phase3_tournament(
            trial_model,
            search_tour_train_loader,
            search_tour_eval_loader,
            epochs=SEARCH_PHASE3_EPOCHS,
            lr=trial_config["lr_phase3"],
            margin=trial_config["margin"],
            trainable_last_blocks=trial_config.get("trainable_last_blocks", 1),
        )

        final_tau = float(h3["mean_tau"][-1]) if h3.get("mean_tau") else float("nan")
        final_val = float(h2["val_loss"][-1]) if h2.get("val_loss") else float("inf")
        best_tau = float(h3["best_mean_tau"][0]) if h3.get("best_mean_tau") else final_tau
        best_val = float(h2["best_val_loss"][0]) if h2.get("best_val_loss") else final_val

        score = (0.0 if np.isnan(best_tau) else best_tau) - 0.05 * best_val

        row = {
            "trial": trial_idx,
            "config": {k: trial_config[k] for k in keys},
            "best_tau": best_tau,
            "best_val_loss": best_val,
            "final_tau": final_tau,
            "final_val_loss": final_val,
            "composite_score": score,
        }
        results.append(row)
        print(
            f"trial_score={score:.4f} | best_tau={best_tau:.4f} | best_val={best_val:.4f} "
            f"| final_tau={final_tau:.4f} | final_val={final_val:.4f}"
        )

        if score > best_score:
            best_score = score
            best_config = dict(trial_config)

    print("\nBest search config:")
    print({k: best_config[k] for k in keys})
    return best_config, results


hparam_results: List[Dict[str, Any]] = []
if RUN_HPARAM_SEARCH:
    # Defaults to first 12 combinations so search remains practical in notebook workflows.
    best_cfg, hparam_results = run_hparam_search(DEFAULT_CONFIG, grid_space, max_trials=12)
    DEFAULT_CONFIG.update(best_cfg)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    hparam_output = CHECKPOINT_DIR / "hparam_results.json"
    with hparam_output.open("w", encoding="utf-8") as f:
        json.dump(hparam_results, f, indent=2)
    print(f"DEFAULT_CONFIG updated from search winner. Saved results to {hparam_output}")
else:
    print("Skipping hyperparameter search (RUN_HPARAM_SEARCH=False).")

# %%
# Execute Curriculum if enabled in Section 0
if RUN_CURRICULUM:
    # 1. Initialize fresh model
    master_model = build_model(DEFAULT_CONFIG)
    histories: Dict[str, Dict[str, List[float]]] = {}
    phase_state_dicts = {
        "init": {k: v.detach().cpu().clone() for k, v in master_model.state_dict().items()}
    }

# %%
# Execute Curriculum if enabled in Section 0
if RUN_CURRICULUM:
    # 2. Phase 1: EDH-only MLM on the supervised corpus
    histories["phase1_mlm"] = train_phase1_mlm(
        master_model,
        sup_mlm_loader,
        epochs=TRAIN_PHASE1_EPOCHS,
        lr=DEFAULT_CONFIG["lr_phase1"],
    )
    phase_state_dicts["phase1"] = {k: v.detach().cpu().clone() for k, v in master_model.state_dict().items()}

# %%
if RUN_CURRICULUM:
    print("\n=== Phase 1 Sanity: EDH MLM Reconstructions ===")
    phase1_sanity = {"train_examples": [], "val_examples": []}

    # Show a couple of examples from both EDH train and validation loaders.
    for split_name, loader in [("train", sup_mlm_loader), ("val", sup_val_loader)]:
        print(f"\n--- {split_name.upper()} SPLIT ---")
        for sample_idx in (0, 1):
            print(f"\n[Sample index: {sample_idx}]")
            ex = show_phase1_mlm_example(
                master_model,
                loader,
                vocab,
                mask_ratio=0.15,
                topk=5,
                sample_index=sample_idx,
                max_display_cards=40,
            )
            phase1_sanity[f"{split_name}_examples"].append(ex)
else:
    print("Skipping Phase 1 sanity check because RUN_CURRICULUM is False.")

# %%
if RUN_CURRICULUM:
    # 3. Phase 2: Supervised Regression
    phase2_best_path = CHECKPOINT_DIR / f"{MASTER_RUN_NAME}_phase2_best.pt"
    histories["phase2_huber"] = train_phase2_huber(
        master_model,
        sup_train_loader,
        sup_val_loader,
        epochs=TRAIN_PHASE2_EPOCHS,
        lr=DEFAULT_CONFIG["lr_phase2"],
        delta=DEFAULT_CONFIG["delta"],
        best_ckpt_path=phase2_best_path,
    )
    phase_state_dicts["phase2"] = {k: v.detach().cpu().clone() for k, v in master_model.state_dict().items()}

# %%
if RUN_CURRICULUM:
    print("\n=== Phase 2 Sanity: Bracket-Stratified Validation Deck Samples ===")
    was_training = master_model.training

    phase2_eval_source = "last_epoch_phase2_weights"
    used_phase2_best_for_sanity = False
    restored_after_phase2_sanity = False

    # Keep Phase 3 training behavior unchanged by restoring the in-memory Phase 2 endpoint after sanity.
    phase2_restore_state = None
    if "phase_state_dicts" in globals() and "phase2" in phase_state_dicts:
        phase2_restore_state = phase_state_dicts["phase2"]

    # Report this block on best Phase 2 checkpoint when available.
    if "phase2_best_path" in globals() and phase2_best_path is not None and Path(phase2_best_path).exists():
        blob = torch.load(phase2_best_path, map_location=device)
        state_dict = blob.get("model_state_dict") if isinstance(blob, dict) else None
        if state_dict is None:
            raise KeyError(f"Checkpoint at {phase2_best_path} is missing model_state_dict")
        master_model.load_state_dict(state_dict)
        phase2_eval_source = f"phase2_best_checkpoint ({phase2_best_path})"
        used_phase2_best_for_sanity = True

    print(f"Using Phase 2 sanity weights: {phase2_eval_source}")
    master_model.eval()

    # Pull records from the validation subset and group by rounded bracket level.
    val_subset = sup_val_loader.dataset
    base_dataset = val_subset.dataset if hasattr(val_subset, "dataset") else val_subset
    val_indices = list(val_subset.indices) if hasattr(val_subset, "indices") else list(range(len(base_dataset)))

    grouped_indices: Dict[int, List[int]] = {}
    for idx in val_indices:
        record = base_dataset.records[int(idx)]
        bracket = int(round(float(record.get("bracket", 3.0))))
        grouped_indices.setdefault(bracket, []).append(int(idx))

    rng = random.Random(SEED)
    stratified_examples = []

    for bracket in sorted(grouped_indices.keys()):
        candidates = grouped_indices[bracket]
        take_n = min(2, len(candidates))
        chosen = rng.sample(candidates, k=take_n) if take_n > 0 else []
        print(f"\n--- Bracket {bracket} ({len(candidates)} decks in val split) ---")

        for chosen_idx in chosen:
            deck_obj = base_dataset.records[chosen_idx]
            triplets = extract_card_qty_role_triplets(deck_obj)
            card_list = [name for name, qty, _ in triplets for _ in range(max(1, int(qty)))]
            target = float(deck_obj.get("bracket", 3.0))

            batch, used_names = deck_to_model_inputs(deck_obj)
            pred = float(master_model(batch["card_ids"], batch["qty_ids"], batch["role_ids"], batch["mask"]).item())

            print(f"pred={pred:.3f} | target={target:.3f} | cards={len(card_list)}")
            print(", ".join(card_list[:40]))
            if len(card_list) > 40:
                print("...")

            stratified_examples.append(
                {
                    "bracket": bracket,
                    "pred": pred,
                    "target": target,
                    "cards": card_list,
                    "used_names": used_names,
                }
            )

    def _collect_preds_and_targets(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        pred_values: List[float] = []
        true_values: List[float] = []
        with torch.no_grad():
            for batch in loader:
                card_ids = batch["card_ids"].to(device)
                qty_ids = batch["qty_ids"].to(device)
                role_ids = batch["role_ids"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].cpu().numpy()
                pred = master_model(card_ids, qty_ids, role_ids, mask).cpu().numpy()
                pred_values.extend([float(x) for x in pred.tolist()])
                true_values.extend([float(x) for x in target.tolist()])
        return np.array(pred_values, dtype=np.float32), np.array(true_values, dtype=np.float32)

    def _compute_regression_metrics(pred_arr: np.ndarray, true_arr: np.ndarray) -> Dict[str, float]:
        if len(pred_arr) == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "num_examples": 0}
        mae = float(np.mean(np.abs(pred_arr - true_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
        pearson = float(np.corrcoef(pred_arr, true_arr)[0, 1]) if len(pred_arr) > 1 else float("nan")
        return {
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "num_examples": int(len(pred_arr)),
        }

    def _mae_by_bracket(pred_arr: np.ndarray, true_arr: np.ndarray) -> Dict[int, float]:
        by_bracket: Dict[int, List[float]] = {}
        for pred, true in zip(pred_arr.tolist(), true_arr.tolist()):
            bracket = int(round(float(true)))
            by_bracket.setdefault(bracket, []).append(abs(float(pred) - float(true)))
        return {b: float(np.mean(vals)) for b, vals in sorted(by_bracket.items())}

    val_pred, val_true = _collect_preds_and_targets(sup_val_loader)
    test_pred, test_true = _collect_preds_and_targets(sup_test_loader)

    val_metrics = _compute_regression_metrics(val_pred, val_true)
    test_metrics = _compute_regression_metrics(test_pred, test_true)
    val_bracket_mae = _mae_by_bracket(val_pred, val_true)
    test_bracket_mae = _mae_by_bracket(test_pred, test_true)

    print("\nValidation metrics")
    print(f"  MAE:     {val_metrics['mae']:.4f}")
    print(f"  RMSE:    {val_metrics['rmse']:.4f}")
    print(f"  Pearson: {val_metrics['pearson']:.4f}")
    print(f"  MAE by rounded bracket: {val_bracket_mae}")

    print("\nHeld-out test metrics")
    print(f"  MAE:     {test_metrics['mae']:.4f}")
    print(f"  RMSE:    {test_metrics['rmse']:.4f}")
    print(f"  Pearson: {test_metrics['pearson']:.4f}")
    print(f"  MAE by rounded bracket: {test_bracket_mae}")

    phase2_sanity = {
        "eval_weight_source": phase2_eval_source,
        "val": {**val_metrics, "mae_by_bracket": val_bracket_mae},
        "test": {**test_metrics, "mae_by_bracket": test_bracket_mae},
        "stratified_examples": stratified_examples,
    }

    # Visual diagnostics and persisted artifacts.
    bracket_rows: List[Dict[str, Any]] = []
    all_brackets = sorted(set(val_bracket_mae.keys()) | set(test_bracket_mae.keys()))
    for b in all_brackets:
        bracket_rows.append(
            {
                "bracket": int(b),
                "val_mae": float(val_bracket_mae.get(b, float("nan"))),
                "test_mae": float(test_bracket_mae.get(b, float("nan"))),
            }
        )

    # Predicted vs true scatter for val/test.
    fig = plt.figure(figsize=(6.5, 6.0))
    plt.scatter(val_true, val_pred, s=10, alpha=0.35, label="val")
    plt.scatter(test_true, test_pred, s=10, alpha=0.35, label="test")
    lo = float(min(val_true.min(initial=1.0), test_true.min(initial=1.0))) if len(val_true) and len(test_true) else 1.0
    hi = float(max(val_true.max(initial=5.0), test_true.max(initial=5.0))) if len(val_true) and len(test_true) else 5.0
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="ideal")
    plt.xlabel("True bracket")
    plt.ylabel("Predicted score")
    plt.title("Phase 2 calibration: predicted vs true")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()
    save_figure_artifact("phase2_pred_vs_true_scatter.png", fig=fig)

    # Residual histogram.
    val_resid = val_pred - val_true
    test_resid = test_pred - test_true
    fig = plt.figure(figsize=(7.5, 4.2))
    plt.hist(val_resid, bins=40, alpha=0.55, label="val residuals")
    plt.hist(test_resid, bins=40, alpha=0.55, label="test residuals")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.1)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title("Phase 2 residual distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    save_figure_artifact("phase2_residual_hist.png", fig=fig)

    # Bracket MAE bars.
    if bracket_rows:
        labels = [str(r["bracket"]) for r in bracket_rows]
        x = np.arange(len(labels), dtype=np.float32)
        width = 0.35

        fig = plt.figure(figsize=(8.0, 4.2))
        plt.bar(x - width / 2, [r["val_mae"] for r in bracket_rows], width=width, label="val")
        plt.bar(x + width / 2, [r["test_mae"] for r in bracket_rows], width=width, label="test")
        plt.xticks(x, labels)
        plt.xlabel("Rounded bracket")
        plt.ylabel("MAE")
        plt.title("Phase 2 MAE by bracket")
        plt.legend()
        plt.grid(axis="y", alpha=0.2)
        plt.show()
        save_figure_artifact("phase2_mae_by_bracket.png", fig=fig)

    phase2_sanity["plot_artifacts"] = {
        "pred_vs_true": str(FIGURES_DIR / "phase2_pred_vs_true_scatter.png"),
        "residual_hist": str(FIGURES_DIR / "phase2_residual_hist.png"),
        "mae_by_bracket": str(FIGURES_DIR / "phase2_mae_by_bracket.png"),
    }

    save_json_artifact("phase2_sanity.json", phase2_sanity)
    save_table_artifact("phase2_mae_by_bracket.csv", bracket_rows)

    if used_phase2_best_for_sanity:
        if phase2_restore_state is not None:
            master_model.load_state_dict(phase2_restore_state)
            restored_after_phase2_sanity = True
            print("Restored in-memory end-of-phase2 weights for subsequent Phase 3 training.")
        else:
            print("[WARN] Could not restore end-of-phase2 weights after sanity; continuing from phase2_best.")

    phase2_sanity["restored_end_of_phase2_weights"] = bool(restored_after_phase2_sanity)

    if was_training:
        master_model.train()
else:
    print("Skipping Phase 2 sanity check because RUN_CURRICULUM is False.")

# %%
if RUN_CURRICULUM:
    # 4. Phase 3: Tournament Calibration
    phase3_best_path = CHECKPOINT_DIR / f"{MASTER_RUN_NAME}_phase3_best.pt"
    histories["phase3_pairwise"] = train_phase3_tournament(
        master_model,
        tour_train_loader,
        tour_eval_loader,
        epochs=TRAIN_PHASE3_EPOCHS,
        lr=DEFAULT_CONFIG["lr_phase3"],
        margin=DEFAULT_CONFIG["margin"],
        trainable_last_blocks=DEFAULT_CONFIG.get("trainable_last_blocks", 1),
        best_ckpt_path=phase3_best_path,
    )
    phase_state_dicts["phase3"] = {k: v.detach().cpu().clone() for k, v in master_model.state_dict().items()}

    # Canonicalize post-training evaluation weights to the best available checkpoint.
    # Preference order is phase3-best, then phase2-best fallback.
    canonical_eval_checkpoint = None
    canonical_eval_source = "last_epoch_in_memory"
    checkpoint_candidates: List[Tuple[str, Path]] = []
    if "phase3_best_path" in globals():
        checkpoint_candidates.append(("phase3_best", phase3_best_path))
    if "phase2_best_path" in globals():
        checkpoint_candidates.append(("phase2_best", phase2_best_path))

    for ckpt_name, ckpt_path in checkpoint_candidates:
        if ckpt_path is None or not Path(ckpt_path).exists():
            continue

        blob = torch.load(ckpt_path, map_location=device)
        state_dict = blob.get("model_state_dict") if isinstance(blob, dict) else None
        if state_dict is None:
            raise KeyError(f"Checkpoint at {ckpt_path} is missing model_state_dict")

        master_model.load_state_dict(state_dict)
        canonical_eval_checkpoint = Path(ckpt_path)
        canonical_eval_source = ckpt_name
        print(f"Loaded canonical eval checkpoint ({ckpt_name}) from {ckpt_path}")
        break

    if canonical_eval_checkpoint is None:
        print("[WARN] No best checkpoint found on disk; using in-memory last-epoch weights for evaluation.")

# %%
if RUN_CURRICULUM:
    print("\n=== Phase 3 Sanity: Tournament Evaluation ===")
    if "canonical_eval_checkpoint" in globals() and canonical_eval_checkpoint is not None:
        print(
            f"Using canonical checkpoint for sanity: {canonical_eval_checkpoint} "
            f"(source={canonical_eval_source})"
        )
    else:
        print("Using in-memory last-epoch weights for sanity.")

    val_tau_stats = compute_tournament_tau_stats(master_model, tour_eval_loader)
    test_tau_stats = compute_tournament_tau_stats(master_model, tour_test_loader)

    print(
        f"Validation Tau mean={val_tau_stats['mean_tau']:.4f}, "
        f"median={val_tau_stats['median_tau']:.4f}, "
        f"positive-corr events={val_tau_stats['positive_corr_pct']:.2f}%"
    )
    print(
        f"Test Tau mean={test_tau_stats['mean_tau']:.4f}, "
        f"median={test_tau_stats['median_tau']:.4f}, "
        f"positive-corr events={test_tau_stats['positive_corr_pct']:.2f}%"
    )

    def _collect_event_taus(model: nn.Module, loader: DataLoader) -> List[float]:
        model.eval()
        taus: List[float] = []
        with torch.no_grad():
            for batch in loader:
                card_ids = batch["card_ids"].to(device)
                qty_ids = batch["qty_ids"].to(device)
                role_ids = batch["role_ids"].to(device)
                mask = batch["mask"].to(device)
                placements = batch["placement"].cpu().numpy()
                if len(placements) < 3:
                    continue

                scores = model(card_ids, qty_ids, role_ids, mask).cpu().numpy()
                tau = _event_tau_for_batch(scores, placements)
                if not np.isnan(tau):
                    taus.append(float(tau))
        return taus

    def _compute_tier_tau_stats(model: nn.Module, loader: DataLoader) -> Dict[str, Dict[str, float]]:
        model.eval()
        by_tier: Dict[str, List[float]] = {}
        with torch.no_grad():
            for batch in loader:
                card_ids = batch["card_ids"].to(device)
                qty_ids = batch["qty_ids"].to(device)
                role_ids = batch["role_ids"].to(device)
                mask = batch["mask"].to(device)
                placements = batch["placement"].cpu().numpy()
                if len(placements) < 3:
                    continue

                scores = model(card_ids, qty_ids, role_ids, mask).cpu().numpy()
                event_tau = _event_tau_for_batch(scores, placements)
                if np.isnan(event_tau):
                    continue

                tier = "unknown"
                if "event_tier" in batch and len(batch["event_tier"]) > 0:
                    tier = str(batch["event_tier"][0])
                by_tier.setdefault(tier, []).append(float(event_tau))

        out: Dict[str, Dict[str, float]] = {}
        for tier, values in by_tier.items():
            arr = np.array(values, dtype=np.float32)
            out[tier] = {
                "events": int(len(arr)),
                "mean_tau": float(arr.mean()),
                "median_tau": float(np.median(arr)),
                "positive_corr_pct": float((arr > 0).mean() * 100.0),
            }
        return out

    val_event_taus = _collect_event_taus(master_model, tour_eval_loader)
    test_event_taus = _collect_event_taus(master_model, tour_test_loader)

    test_tau_by_tier = _compute_tier_tau_stats(master_model, tour_test_loader)
    if test_tau_by_tier:
        print("\nTest Tau by tier")
        for tier, metrics in sorted(test_tau_by_tier.items()):
            print(
                f"  {tier}: events={metrics['events']}, "
                f"mean={metrics['mean_tau']:.4f}, median={metrics['median_tau']:.4f}, "
                f"positive={metrics['positive_corr_pct']:.2f}%"
            )

    was_training = master_model.training
    master_model.eval()
    phase3_examples = []

    with torch.no_grad():
        for event_idx, batch in enumerate(tour_eval_loader):
            if event_idx >= 3:
                break

            card_ids = batch["card_ids"].to(device)
            qty_ids = batch["qty_ids"].to(device)
            role_ids = batch["role_ids"].to(device)
            mask = batch["mask"].to(device)
            placements = batch["placement"].cpu().numpy()
            scores = master_model(card_ids, qty_ids, role_ids, mask).cpu().numpy()

            pred_order = np.argsort(-scores).tolist()
            actual_order = np.argsort(placements).tolist()
            event_tau = _event_tau_for_batch(scores, placements)

            example = {
                "event_index": int(event_idx),
                "num_decks": int(len(scores)),
                "tau": float(event_tau),
                "pred_order": pred_order,
                "actual_order": actual_order,
                "scores": [float(x) for x in scores.tolist()],
                "placements": [float(x) for x in placements.tolist()],
            }
            phase3_examples.append(example)

            print(f"\nEvent {event_idx} | decks={len(scores)} | tau={event_tau:.4f}")
            print(f"Pred order (best->worst):   {pred_order}")
            print(f"Actual order (best->worst): {actual_order}")

    tier_rows: List[Dict[str, Any]] = []
    for tier, metrics in sorted(test_tau_by_tier.items()):
        tier_rows.append(
            {
                "tier": str(tier),
                "events": int(metrics["events"]),
                "mean_tau": float(metrics["mean_tau"]),
                "median_tau": float(metrics["median_tau"]),
                "positive_corr_pct": float(metrics["positive_corr_pct"]),
            }
        )

    # Tau distribution plot.
    fig = plt.figure(figsize=(7.5, 4.4))
    if val_event_taus:
        plt.hist(val_event_taus, bins=20, alpha=0.55, label="val event tau")
    if test_event_taus:
        plt.hist(test_event_taus, bins=20, alpha=0.55, label="test event tau")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.1)
    plt.xlabel("Kendall tau")
    plt.ylabel("Event count")
    plt.title("Phase 3 event tau distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    save_figure_artifact("phase3_tau_hist.png", fig=fig)

    # Tier-wise mean tau bars.
    if tier_rows:
        fig = plt.figure(figsize=(8.0, 4.2))
        tiers = [row["tier"] for row in tier_rows]
        mean_taus = [row["mean_tau"] for row in tier_rows]
        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in mean_taus]
        plt.bar(np.arange(len(tiers)), mean_taus, color=colors)
        plt.xticks(np.arange(len(tiers)), tiers, rotation=30, ha="right")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        plt.xlabel("Event tier")
        plt.ylabel("Mean tau")
        plt.title("Phase 3 mean tau by tier (test)")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.show()
        save_figure_artifact("phase3_tau_by_tier.png", fig=fig)

    phase3_sanity = {
        "val_tau_stats": val_tau_stats,
        "test_tau_stats": test_tau_stats,
        "val_event_taus": [float(x) for x in val_event_taus],
        "test_event_taus": [float(x) for x in test_event_taus],
        "test_tau_by_tier": test_tau_by_tier,
        "examples": phase3_examples,
        "plot_artifacts": {
            "tau_hist": str(FIGURES_DIR / "phase3_tau_hist.png"),
            "tau_by_tier": str(FIGURES_DIR / "phase3_tau_by_tier.png"),
        },
    }

    save_json_artifact("phase3_sanity.json", phase3_sanity)
    save_table_artifact("phase3_tau_by_tier.csv", tier_rows)

    if was_training:
        master_model.train()
else:
    print("Skipping Phase 3 sanity check because RUN_CURRICULUM is False.")

# %%
def _forward_with_self_attn(
    model: SetTransformer,
    card_ids: torch.Tensor,
    qty_ids: torch.Tensor,
    role_ids: torch.Tensor,
    mask: torch.Tensor,
) -> List[torch.Tensor]:
    """Run encoder manually and capture per-layer self-attention maps [heads, seq, seq]."""
    x = model.embed_tokens(card_ids, qty_ids, role_ids)
    attn_maps: List[torch.Tensor] = []

    for layer in model.transformer_encoder.layers:
        if layer.norm_first:
            x_norm = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=~mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attn_out)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            attn_out, attn_w = layer.self_attn(
                x,
                x,
                x,
                key_padding_mask=~mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            x = layer.norm2(x + layer._ff_block(x))

        # attn_w shape: [batch, heads, seq, seq]
        attn_maps.append(attn_w[0].detach().cpu())

    return attn_maps

def _load_phase_model(phase_name: str) -> SetTransformer:
    if "phase_state_dicts" not in globals() or phase_name not in phase_state_dicts:
        raise KeyError(f"Missing phase snapshot: {phase_name}")
    m = build_model(DEFAULT_CONFIG)
    m.load_state_dict(phase_state_dicts[phase_name])
    m.eval()
    return m

def visualize_combo_attention_across_phases(
    deck_obj: Dict[str, Any],
    combo_cards: List[str],
    phases: List[str] = ["init", "phase1", "phase2", "phase3"],
    layer_idx: int = -1,
    head_reduce: str = "mean",
) -> Dict[str, Any]:
    batch, used_names = deck_to_model_inputs(deck_obj)
    name_to_pos = {name: i for i, name in enumerate(used_names)}
    combo_norm = [normalize_card_name(c) for c in combo_cards]
    combo_pos = [name_to_pos[c] for c in combo_norm if c in name_to_pos]

    if len(combo_pos) < 2:
        raise ValueError("Need at least two combo cards present in the sampled deck.")

    out: Dict[str, Any] = {}
    for ph in phases:
        phase_model = _load_phase_model(ph)
        maps = _forward_with_self_attn(
            phase_model,
            batch["card_ids"],
            batch["qty_ids"],
            batch["role_ids"],
            batch["mask"],
        )
        layer_map = maps[layer_idx]  # [heads, seq, seq]
        if head_reduce == "max":
            reduced = layer_map.max(dim=0).values.numpy()
        else:
            reduced = layer_map.mean(dim=0).numpy()

        combo_mat = reduced[np.ix_(combo_pos, combo_pos)]
        out[ph] = combo_mat

        plt.figure(figsize=(4, 3))
        plt.imshow(combo_mat, cmap="magma")
        plt.colorbar()
        labels = [used_names[p] for p in combo_pos]
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title(f"Self-attention combo map | {ph}")
        plt.tight_layout()
        plt.show()

    return out

def token_movement_report(
    card_names: List[str],
    phases: List[str] = ["init", "phase1", "phase2", "phase3"],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_model = _load_phase_model("init")
    base_w = base_model.card_embedding.weight.detach().cpu()

    for card in card_names:
        norm = normalize_card_name(card)
        cid = vocab.get(norm)
        if cid is None:
            rows.append({"card": card, "status": "missing_vocab"})
            continue

        base_vec = base_w[cid].numpy()
        row: Dict[str, Any] = {"card": norm}
        for ph in phases:
            ph_model = _load_phase_model(ph)
            vec = ph_model.card_embedding.weight.detach().cpu()[cid].numpy()
            denom = (np.linalg.norm(base_vec) * np.linalg.norm(vec)) + 1e-12
            cos = float(np.dot(base_vec, vec) / denom)
            l2 = float(np.linalg.norm(vec - base_vec))
            row[f"cos_to_init_{ph}"] = cos
            row[f"l2_from_init_{ph}"] = l2
        rows.append(row)

    for r in rows:
        print(r)
    return rows

def embedding_cosine_sanity(
    anchor_cards: List[str],
    topk: int = 5,
    embedding_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if embedding_weights is None:
        embedding_weights = master_model.card_embedding.weight.detach().cpu() if "master_model" in globals() else weights.detach().cpu()

    present = []
    missing = []
    for c in anchor_cards:
        norm = normalize_card_name(c)
        if norm in vocab:
            present.append(norm)
        else:
            missing.append(norm)

    print(f"Present anchors: {len(present)} | Missing anchors: {len(missing)}")
    if missing:
        print("Missing:", missing)

    id_to_card_local = {idx: card for card, idx in vocab.items()}
    sim_examples = {}
    for c in present:
        query_idx = vocab[c]
        query_vec = embedding_weights[query_idx].detach().cpu().numpy()
        query_den = np.linalg.norm(query_vec) + 1e-12

        sims = []
        for idx in range(3, embedding_weights.shape[0]):
            cand_name = id_to_card_local.get(idx, f"<{idx}>")
            if cand_name == c:
                continue
            cand_vec = embedding_weights[idx].detach().cpu().numpy()
            denom = query_den * (np.linalg.norm(cand_vec) + 1e-12)
            sim = float(np.dot(query_vec, cand_vec) / denom)
            sims.append((cand_name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        sim_examples[c] = sims[:topk]
        print(f"\nNearest neighbors for {c}:")
        for name, sim in sim_examples[c]:
            print(f"  {name}: {sim:.4f}")

    if len(present) >= 2:
        idxs = [vocab[c] for c in present]
        mat = embedding_weights[idxs].detach().cpu().numpy()
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        cos_mat = mat @ mat.T
        plt.figure(figsize=(max(4, len(present) * 0.7), max(3, len(present) * 0.7)))
        plt.imshow(cos_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        plt.colorbar()
        plt.xticks(range(len(present)), present, rotation=45, ha="right")
        plt.yticks(range(len(present)), present)
        plt.title("Anchor cosine similarity matrix")
        plt.tight_layout()
        plt.show()
    else:
        cos_mat = None

    return {"present": present, "missing": missing, "neighbors": sim_examples, "cosine_matrix": cos_mat}

if RUN_CURRICULUM:
    print("\n=== Extra Diagnostics: Attention, Token Movement, Embedding Cosine ===")
    # Pick one stratified val example as reference deck for attention tracing.
    if "phase2_sanity" in globals() and phase2_sanity.get("stratified_examples"):
        ref_deck_cards = phase2_sanity["stratified_examples"][0]["cards"]
        ref_deck_obj = {"cards": {c: 1 for c in ref_deck_cards}}
    else:
        ref_deck_obj = sup_dataset.records[0]

    combo_candidates = ["thassa's oracle", "demonic consultation", "tainted pact", "underworld breach"]
    try:
        phase_attention_maps = visualize_combo_attention_across_phases(ref_deck_obj, combo_candidates)
    except Exception as exc:
        print(f"Combo attention visualization skipped: {exc}")

    moved_card_report = token_movement_report(combo_candidates)
    embedding_sanity = embedding_cosine_sanity(combo_candidates, topk=6)

# %%
@torch.no_grad()
def collect_raw_and_bounded_preds(model: nn.Module, loader: DataLoader):
    model.eval()
    raw_vals: List[float] = []
    bounded_vals: List[float] = []
    true_vals: List[float] = []

    for batch in loader:
        card_ids = batch["card_ids"].to(device)
        qty_ids = batch["qty_ids"].to(device)
        role_ids = batch["role_ids"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].cpu().numpy()

        raw = model.forward_raw(card_ids, qty_ids, role_ids, mask).cpu().numpy()
        bounded = model(card_ids, qty_ids, role_ids, mask).cpu().numpy()

        raw_vals.extend([float(x) for x in raw.tolist()])
        bounded_vals.extend([float(x) for x in bounded.tolist()])
        true_vals.extend([float(x) for x in target.tolist()])

    return (
        np.array(raw_vals, dtype=np.float32),
        np.array(bounded_vals, dtype=np.float32),
        np.array(true_vals, dtype=np.float32),
    )


def fit_phase2_isotonic_calibrator(model: nn.Module, val_loader: DataLoader, save_path: Path):
    from sklearn.isotonic import IsotonicRegression
    import joblib

    raw_val, bounded_val, y_val = collect_raw_and_bounded_preds(model, val_loader)

    calibrator = IsotonicRegression(
        increasing=True,
        y_min=1.0,
        y_max=5.0,
        out_of_bounds="clip",
    )
    calibrator.fit(raw_val, y_val)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, save_path)

    calibrated_val = calibrator.predict(raw_val)
    summary = {
        "val_uncalibrated_mae": float(np.mean(np.abs(bounded_val - y_val))),
        "val_calibrated_mae": float(np.mean(np.abs(calibrated_val - y_val))),
        "val_uncalibrated_rmse": float(np.sqrt(np.mean((bounded_val - y_val) ** 2))),
        "val_calibrated_rmse": float(np.sqrt(np.mean((calibrated_val - y_val) ** 2))),
        "save_path": str(save_path),
    }
    return calibrator, summary


if RUN_CURRICULUM:
    # 5. Plot all phase curves and persist them.
    training_curve_artifacts = plot_training_curves(histories, save=True)

    # Ensure final export uses canonical best-available weights.
    if "canonical_eval_checkpoint" in globals() and canonical_eval_checkpoint is not None and Path(canonical_eval_checkpoint).exists():
        blob = torch.load(canonical_eval_checkpoint, map_location=device)
        state_dict = blob.get("model_state_dict") if isinstance(blob, dict) else None
        if state_dict is None:
            raise KeyError(f"Checkpoint at {canonical_eval_checkpoint} is missing model_state_dict")
        master_model.load_state_dict(state_dict)

    # Fit post-hoc calibrator on validation raw scores from the canonical model.
    score_calibrator = None
    calibrator_path = None
    calibration_summary = None
    calibration_summary_path = None
    if "sup_val_loader" in globals():
        calibrator_path = CHECKPOINT_DIR / f"{MASTER_RUN_NAME}_isotonic_calibrator.joblib"
        score_calibrator, calibration_summary = fit_phase2_isotonic_calibrator(
            master_model,
            sup_val_loader,
            calibrator_path,
        )
        print("Calibration summary:", calibration_summary)
        calibration_summary_path = save_json_artifact("phase2_calibration_summary.json", calibration_summary)

    # 6. Save canonical final weights
    final_path = CHECKPOINT_DIR / f"{MASTER_RUN_NAME}.pt"
    artifact_paths = {
        "final": str(final_path),
        "results_dir": str(RESULTS_DIR),
        "figures_dir": str(FIGURES_DIR),
        "tables_dir": str(TABLES_DIR),
    }

    if "phase2_best_path" in globals():
        artifact_paths["phase2_best"] = str(phase2_best_path)
    if "phase3_best_path" in globals():
        artifact_paths["phase3_best"] = str(phase3_best_path)
    if "canonical_eval_checkpoint" in globals() and canonical_eval_checkpoint is not None:
        artifact_paths["canonical_eval_checkpoint"] = str(canonical_eval_checkpoint)
        artifact_paths["canonical_eval_source"] = str(canonical_eval_source)
    if calibrator_path is not None:
        artifact_paths["score_calibrator"] = str(calibrator_path)
    if calibration_summary_path is not None:
        artifact_paths["result_phase2_calibration_summary.json"] = str(calibration_summary_path)

    if isinstance(training_curve_artifacts, list):
        for i, path in enumerate(training_curve_artifacts):
            artifact_paths[f"training_curve_{i}"] = str(path)

    # Record known analysis artifacts when they are available.
    known_result_files = [
        "phase2_sanity.json",
        "phase3_sanity.json",
        "benchmark_summary.json",
        "phase2_calibration_summary.json",
    ]
    known_table_files = [
        "phase2_mae_by_bracket.csv",
        "phase3_tau_by_tier.csv",
        "benchmark_results.csv",
    ]
    known_figure_files = [
        "phase2_pred_vs_true_scatter.png",
        "phase2_residual_hist.png",
        "phase2_mae_by_bracket.png",
        "phase3_tau_hist.png",
        "phase3_tau_by_tier.png",
        "benchmark_pred_vs_expected.png",
        "benchmark_pass_fail.png",
    ]

    for fname in known_result_files:
        path = RESULTS_DIR / fname
        if path.exists():
            artifact_paths[f"result_{fname}"] = str(path)
    for fname in known_table_files:
        path = TABLES_DIR / fname
        if path.exists():
            artifact_paths[f"table_{fname}"] = str(path)
    for fname in known_figure_files:
        path = FIGURES_DIR / fname
        if path.exists():
            artifact_paths[f"figure_{fname}"] = str(path)

    torch.save(
        {
            "model_state_dict": master_model.state_dict(),
            "config": DEFAULT_CONFIG,
            "vocab_size": VOCAB_SIZE,
            "max_qty": MAX_QTY_EMBED,
            "histories": histories,
            "artifact_paths": artifact_paths,
        },
        final_path,
    )

    save_json_artifact("artifact_manifest.json", artifact_paths)

    print(f"\nMaster Curriculum Complete! Final model saved to {final_path}")
    if "phase2_best_path" in globals():
        print(f"Best Phase 2 checkpoint: {phase2_best_path}")
    if "phase3_best_path" in globals():
        print(f"Best Phase 3 checkpoint: {phase3_best_path}")
    if "canonical_eval_checkpoint" in globals() and canonical_eval_checkpoint is not None:
        print(
            f"Canonical evaluation/export checkpoint: {canonical_eval_checkpoint} "
            f"(source={canonical_eval_source})"
        )
    if calibrator_path is not None:
        print(f"Score calibrator saved to {calibrator_path}")
    print(f"Artifact manifest saved to {RESULTS_DIR / 'artifact_manifest.json'}")

# %% [markdown]
# ## 5. Inspection, Visualization, and Sanity Checks

# %%
# Static embedding-only utilities were moved to card_embedding_analysis.ipynb.
# This section keeps only deck-grading diagnostics tied to the trained model.

@torch.no_grad()
def score_deck(
    model: nn.Module,
    deck_obj: Dict[str, Any],
    calibrator=None,
    return_components: bool = False,
):
    model.eval()
    batch, _ = deck_to_model_inputs(deck_obj)

    raw_score = float(
        model.forward_raw(
            batch["card_ids"],
            batch["qty_ids"],
            batch["role_ids"],
            batch["mask"],
        ).item()
    )
    bounded_score = float(
        model(
            batch["card_ids"],
            batch["qty_ids"],
            batch["role_ids"],
            batch["mask"],
        ).item()
    )

    calibrated_score = float(calibrator.predict([raw_score])[0]) if calibrator is not None else bounded_score

    if return_components:
        return {
            "raw_score": raw_score,
            "bounded_score": bounded_score,
            "calibrated_score": calibrated_score,
        }

    return calibrated_score


@torch.no_grad()
def inspect_deck_attention(model: nn.Module, deck_obj: Dict[str, Any], topk: int = 20):
    model.eval()
    batch, used_names = deck_to_model_inputs(deck_obj)
    score, attn = model(
        batch["card_ids"],
        batch["qty_ids"],
        batch["role_ids"],
        batch["mask"],
        return_attention=True,
    )

    attn = attn.squeeze(0).detach().cpu().numpy()  # [heads, num_seeds, seq_len]
    mean_token_attention = attn.mean(axis=(0, 1))
    token_scores = list(zip(used_names, mean_token_attention[:len(used_names)].tolist()))
    token_scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "score": float(score.item()),
        "top_attention": token_scores[:topk],
    }


def plot_attention_bar(attention_summary: Dict[str, Any], topk: int = 15):
    items = attention_summary["top_attention"][:topk]
    labels = [x[0] for x in items][::-1]
    values = [x[1] for x in items][::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, values)
    plt.title(f"Top PMA Attention Cards | Deck Score {attention_summary['score']:.2f}")
    plt.xlabel("Mean attention weight")
    plt.show()


def run_sanity_suite(
    model: nn.Module,
    named_decks: Dict[str, Dict[str, Any]],
    expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    calibrator=None,
) -> List[Dict[str, Any]]:
    rows = []
    for name, deck in named_decks.items():
        pred = score_deck(model, deck, calibrator=calibrator)
        row = {"deck": name, "predicted_score": pred}
        if expected_ranges and name in expected_ranges:
            lo, hi = expected_ranges[name]
            row["expected_range"] = (lo, hi)
            row["within_range"] = lo <= pred <= hi
        rows.append(row)

    for row in rows:
        print(row)
    return rows


# These ranges are initial expectations for fixed reference decks below and should be
# tightened after a few stable training runs.
SANITY_EXPECTED_RANGES = {
    "cedh_blue_farm_reference": (4.6, 5.0),
    "low_bracket_reference_supervised": (1.8, 3.0),
    "gameknights_episode84_reference": (2.8, 3.9),
    "trap_no_lands_full_100": (1.0, 2.4),
}

# %%
def _resolve_subset_records(dataset_obj):
    base_dataset = dataset_obj.dataset if hasattr(dataset_obj, "dataset") else dataset_obj
    indices = list(dataset_obj.indices) if hasattr(dataset_obj, "indices") else list(range(len(base_dataset)))
    return base_dataset, indices


def _clamp_bracket(value: float) -> int:
    return max(1, min(5, int(round(float(value)))))


def _extract_supervised_test_smoke_rows(
    test_dataset,
    model: nn.Module,
    calibrator=None,
    per_bracket: int = 8,
    seed: int = 42,
    max_preview_cards: int = 30,
) -> List[Dict[str, Any]]:
    base_dataset, indices = _resolve_subset_records(test_dataset)

    grouped: Dict[int, List[int]] = {b: [] for b in range(1, 6)}
    for idx in indices:
        record = base_dataset.records[int(idx)]
        bracket = _clamp_bracket(record.get("bracket", 3.0))
        grouped[bracket].append(int(idx))

    rng = random.Random(seed)
    selected_indices: List[int] = []
    for bracket in range(1, 6):
        candidates = grouped.get(bracket, [])
        if not candidates:
            continue
        take_n = min(int(per_bracket), len(candidates))
        selected_indices.extend(rng.sample(candidates, k=take_n))

    rows: List[Dict[str, Any]] = []
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for idx in selected_indices:
            record = base_dataset.records[int(idx)]
            triplets = extract_card_qty_role_triplets(record)

            cards: Dict[str, int] = {}
            for name, qty, _ in triplets:
                cards[str(name)] = cards.get(str(name), 0) + max(1, int(qty))

            label = float(record.get("bracket", 3.0))
            label_round = _clamp_bracket(label)
            deck_id = str(record.get("deck_id", record.get("deck_url", f"test_idx_{idx}")))
            score_parts = score_deck(
                model,
                {"cards": cards},
                calibrator=calibrator,
                return_components=True,
            )

            raw_score = float(score_parts["raw_score"])
            bounded_score = float(score_parts["bounded_score"])
            calibrated_score = float(score_parts["calibrated_score"])

            preview = [f"{name} x{qty}" for name, qty in sorted(cards.items())[:max_preview_cards]]
            rows.append(
                {
                    "split": "test",
                    "dataset_index": int(idx),
                    "deck_id": deck_id,
                    "label_bracket": label,
                    "label_bracket_round": int(label_round),
                    "raw_score": raw_score,
                    "bounded_score": bounded_score,
                    "calibrated_score": calibrated_score,
                    "predicted_score": calibrated_score,
                    "abs_error": float(abs(calibrated_score - label)),
                    "abs_error_uncalibrated": float(abs(bounded_score - label)),
                    "num_unique_cards": int(len(cards)),
                    "num_total_cards": int(sum(cards.values())),
                    "cards": cards,
                    "cards_preview": preview,
                }
            )

    if model_was_training:
        model.train()

    rows.sort(key=lambda r: (r["label_bracket_round"], r["deck_id"]))
    return rows


def _smoke_diagnostics(rows: List[Dict[str, Any]], score_key: str = "predicted_score") -> Dict[str, Any]:
    if not rows:
        return {
            "overall": {"count": 0},
            "per_bracket": {},
            "confusion_matrix": [],
            "mean_baseline": {},
            "score_key": score_key,
        }

    true_vals = np.array([float(r["label_bracket"]) for r in rows], dtype=np.float32)
    pred_vals = np.array([float(r[score_key]) for r in rows], dtype=np.float32)

    per_bracket: Dict[int, Dict[str, float]] = {}
    for b in range(1, 6):
        mask = np.array([int(r["label_bracket_round"]) == b for r in rows], dtype=bool)
        if not np.any(mask):
            per_bracket[b] = {
                "count": 0,
                "true_mean": float("nan"),
                "pred_mean": float("nan"),
                "pred_std": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
            }
            continue

        t = true_vals[mask]
        p = pred_vals[mask]
        per_bracket[b] = {
            "count": int(mask.sum()),
            "true_mean": float(t.mean()),
            "pred_mean": float(p.mean()),
            "pred_std": float(p.std()),
            "mae": float(np.mean(np.abs(p - t))),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        }

    overall = {
        "count": int(len(rows)),
        "true_mean": float(true_vals.mean()),
        "pred_mean": float(pred_vals.mean()),
        "pred_std": float(pred_vals.std()),
        "mae": float(np.mean(np.abs(pred_vals - true_vals))),
        "rmse": float(np.sqrt(np.mean((pred_vals - true_vals) ** 2))),
        "pearson": float(np.corrcoef(pred_vals, true_vals)[0, 1]) if len(rows) > 1 else float("nan"),
    }

    mean_baseline_pred = float(true_vals.mean())
    baseline = np.full_like(true_vals, mean_baseline_pred)
    mean_baseline = {
        "prediction": mean_baseline_pred,
        "mae": float(np.mean(np.abs(baseline - true_vals))),
        "rmse": float(np.sqrt(np.mean((baseline - true_vals) ** 2))),
    }

    true_round = np.array([_clamp_bracket(v) for v in true_vals.tolist()], dtype=np.int32)
    pred_round = np.array([_clamp_bracket(v) for v in pred_vals.tolist()], dtype=np.int32)
    cm = np.zeros((5, 5), dtype=np.int32)
    for t, p in zip(true_round.tolist(), pred_round.tolist()):
        cm[t - 1, p - 1] += 1

    return {
        "overall": overall,
        "per_bracket": per_bracket,
        "confusion_matrix": cm.tolist(),
        "mean_baseline": mean_baseline,
        "score_key": score_key,
    }


if "sup_test_loader" in globals() and "master_model" in globals():
    active_calibrator = score_calibrator if "score_calibrator" in globals() else None

    test_smoke_rows = _extract_supervised_test_smoke_rows(
        sup_test_loader.dataset,
        model=master_model,
        calibrator=active_calibrator,
        per_bracket=8,
        seed=SEED,
        max_preview_cards=30,
    )

    label_counts: Dict[int, int] = {b: 0 for b in range(1, 6)}
    for row in test_smoke_rows:
        label_counts[int(row["label_bracket_round"])] += 1

    print(f"Collected {len(test_smoke_rows)} supervised test decks with labels and predictions.")
    print(f"Rounded label coverage (1-5): {label_counts}")
    missing = [b for b, n in label_counts.items() if n == 0]
    if missing:
        print(f"[WARN] No decks found for bracket(s): {missing}")

    for row in test_smoke_rows[:8]:
        print(
            {
                "deck_id": row["deck_id"],
                "label_bracket": row["label_bracket"],
                "bounded_score": row["bounded_score"],
                "calibrated_score": row["calibrated_score"],
                "abs_error_uncalibrated": row["abs_error_uncalibrated"],
                "abs_error": row["abs_error"],
            }
        )

    smoke_diag_bounded = _smoke_diagnostics(test_smoke_rows, score_key="bounded_score")
    smoke_diag_calibrated = _smoke_diagnostics(test_smoke_rows, score_key="calibrated_score")

    print("\nPer-bracket prediction stats (bounded)")
    per_bracket_rows: List[Dict[str, Any]] = []
    for b in range(1, 6):
        b_stats = smoke_diag_bounded["per_bracket"].get(b, {})
        c_stats = smoke_diag_calibrated["per_bracket"].get(b, {})
        row = {
            "true_bracket_round": b,
            "count": int(c_stats.get("count", 0)),
            "bounded_mean": float(b_stats.get("pred_mean", float("nan"))),
            "bounded_std": float(b_stats.get("pred_std", float("nan"))),
            "calibrated_mean": float(c_stats.get("pred_mean", float("nan"))),
            "calibrated_std": float(c_stats.get("pred_std", float("nan"))),
            "bounded_mae": float(b_stats.get("mae", float("nan"))),
            "calibrated_mae": float(c_stats.get("mae", float("nan"))),
        }
        per_bracket_rows.append(row)
        print(row)

    print("\nOverall vs mean-baseline")
    print(
        {
            "bounded": smoke_diag_bounded["overall"],
            "calibrated": smoke_diag_calibrated["overall"],
            "mean_baseline": smoke_diag_calibrated["mean_baseline"],
        }
    )

    save_json_artifact("supervised_test_smoke_rows.json", test_smoke_rows)

    table_rows = [
        {
            "split": row["split"],
            "dataset_index": row["dataset_index"],
            "deck_id": row["deck_id"],
            "label_bracket": row["label_bracket"],
            "label_bracket_round": row["label_bracket_round"],
            "raw_score": row["raw_score"],
            "bounded_score": row["bounded_score"],
            "calibrated_score": row["calibrated_score"],
            "abs_error_uncalibrated": row["abs_error_uncalibrated"],
            "abs_error": row["abs_error"],
            "num_unique_cards": row["num_unique_cards"],
            "num_total_cards": row["num_total_cards"],
        }
        for row in test_smoke_rows
    ]
    save_table_artifact("supervised_test_smoke_rows.csv", table_rows)

    # Overall score histogram before/after calibration.
    bounded_vals = np.array([float(r["bounded_score"]) for r in test_smoke_rows], dtype=np.float32)
    calibrated_vals = np.array([float(r["calibrated_score"]) for r in test_smoke_rows], dtype=np.float32)

    fig = plt.figure(figsize=(7.8, 4.4))
    plt.hist(bounded_vals, bins=20, alpha=0.45, label="bounded")
    plt.hist(calibrated_vals, bins=20, alpha=0.45, label="calibrated")
    plt.xlabel("Predicted score")
    plt.ylabel("Count")
    plt.title("Smoke test: overall prediction histogram")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    save_figure_artifact("supervised_test_smoke_pred_hist_overall.png", fig=fig)

    # Histogram by true bracket for bounded score.
    fig = plt.figure(figsize=(8.4, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for b, c in zip(range(1, 6), colors):
        vals = [float(r["bounded_score"]) for r in test_smoke_rows if int(r["label_bracket_round"]) == b]
        if vals:
            plt.hist(vals, bins=12, alpha=0.35, label=f"true={b}", color=c)
    plt.xlabel("Bounded score")
    plt.ylabel("Count")
    plt.title("Smoke test: bounded-score histogram by true bracket")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    save_figure_artifact("supervised_test_smoke_hist_by_bracket_bounded.png", fig=fig)

    # Histogram by true bracket for calibrated score.
    fig = plt.figure(figsize=(8.4, 4.6))
    for b, c in zip(range(1, 6), colors):
        vals = [float(r["calibrated_score"]) for r in test_smoke_rows if int(r["label_bracket_round"]) == b]
        if vals:
            plt.hist(vals, bins=12, alpha=0.35, label=f"true={b}", color=c)
    plt.xlabel("Calibrated score")
    plt.ylabel("Count")
    plt.title("Smoke test: calibrated-score histogram by true bracket")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
    save_figure_artifact("supervised_test_smoke_hist_by_bracket_calibrated.png", fig=fig)

    # Rounded confusion matrix heatmap (rows=true, cols=pred), bounded and calibrated.
    cm_bounded = np.array(smoke_diag_bounded["confusion_matrix"], dtype=np.int32)
    fig = plt.figure(figsize=(5.6, 4.8))
    plt.imshow(cm_bounded, cmap="Blues")
    plt.colorbar(label="Count")
    ticks = np.arange(5)
    labels = ["1", "2", "3", "4", "5"]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted bracket (rounded)")
    plt.ylabel("True bracket (rounded)")
    plt.title("Smoke confusion matrix (bounded)")
    for i in range(cm_bounded.shape[0]):
        for j in range(cm_bounded.shape[1]):
            plt.text(j, i, str(int(cm_bounded[i, j])), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()
    save_figure_artifact("supervised_test_smoke_confusion_matrix_bounded.png", fig=fig)

    cm_calibrated = np.array(smoke_diag_calibrated["confusion_matrix"], dtype=np.int32)
    fig = plt.figure(figsize=(5.6, 4.8))
    plt.imshow(cm_calibrated, cmap="Greens")
    plt.colorbar(label="Count")
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted bracket (rounded)")
    plt.ylabel("True bracket (rounded)")
    plt.title("Smoke confusion matrix (calibrated)")
    for i in range(cm_calibrated.shape[0]):
        for j in range(cm_calibrated.shape[1]):
            plt.text(j, i, str(int(cm_calibrated[i, j])), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()
    save_figure_artifact("supervised_test_smoke_confusion_matrix_calibrated.png", fig=fig)

    confusion_bounded_rows: List[Dict[str, Any]] = []
    confusion_calibrated_rows: List[Dict[str, Any]] = []
    for i in range(5):
        confusion_bounded_rows.append(
            {
                "true_bracket": i + 1,
                "pred_1": int(cm_bounded[i, 0]),
                "pred_2": int(cm_bounded[i, 1]),
                "pred_3": int(cm_bounded[i, 2]),
                "pred_4": int(cm_bounded[i, 3]),
                "pred_5": int(cm_bounded[i, 4]),
            }
        )
        confusion_calibrated_rows.append(
            {
                "true_bracket": i + 1,
                "pred_1": int(cm_calibrated[i, 0]),
                "pred_2": int(cm_calibrated[i, 1]),
                "pred_3": int(cm_calibrated[i, 2]),
                "pred_4": int(cm_calibrated[i, 3]),
                "pred_5": int(cm_calibrated[i, 4]),
            }
        )

    save_table_artifact("supervised_test_smoke_per_bracket_stats.csv", per_bracket_rows)
    save_table_artifact("supervised_test_smoke_confusion_matrix_bounded.csv", confusion_bounded_rows)
    save_table_artifact("supervised_test_smoke_confusion_matrix_calibrated.csv", confusion_calibrated_rows)

    save_json_artifact(
        "supervised_test_smoke_summary.json",
        {
            "total_rows": int(len(test_smoke_rows)),
            "per_bracket_counts": label_counts,
            "target_brackets": [1, 2, 3, 4, 5],
            "per_bracket_target": 8,
            "diagnostics_bounded": smoke_diag_bounded,
            "diagnostics_calibrated": smoke_diag_calibrated,
            "plot_artifacts": {
                "overall_hist": str(FIGURES_DIR / "supervised_test_smoke_pred_hist_overall.png"),
                "hist_by_bracket_bounded": str(FIGURES_DIR / "supervised_test_smoke_hist_by_bracket_bounded.png"),
                "hist_by_bracket_calibrated": str(FIGURES_DIR / "supervised_test_smoke_hist_by_bracket_calibrated.png"),
                "confusion_matrix_bounded": str(FIGURES_DIR / "supervised_test_smoke_confusion_matrix_bounded.png"),
                "confusion_matrix_calibrated": str(FIGURES_DIR / "supervised_test_smoke_confusion_matrix_calibrated.png"),
            },
        },
    )
else:
    print("Skipping supervised test smoke export because sup_test_loader or master_model is unavailable.")


