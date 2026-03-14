import itertools
import json
import re
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset, random_split


EMBEDDING_MODEL = Path("../../data/general-item2vec_mtg.model")
INPUT_EDH = Path("../../data/edh-decks.jsonl")

RESULTS_OUTPUT = Path("hyperparameter_results.json")
BEST_WEIGHTS_OUTPUT = Path("best_model_weights.pth")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

SEED = 42
MAX_LEN = 115
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMS = [64, 128, 256]
CLASSIFIER_LRS = [3e-3, 1e-3, 3e-4]
DROPOUTS = [0.2, 0.3]
BATCH_SIZES = [64]
WEIGHT_DECAYS = [1e-4]
EMBEDDING_LR = 1e-5
EPOCHS_PER_TRIAL = 10

BASIC_LANDS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest"
}

BRACKET_RE = re.compile(r"bracket[-_ ]?([1-5])")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gensim_embeddings(gensim_model_path: Path):
    print(f"Loading embeddings from {gensim_model_path}...")
    g_model = Word2Vec.load(str(gensim_model_path))

    gensim_words = g_model.wv.index_to_key
    vector_size = g_model.vector_size
    vocab_size = len(gensim_words) + 2

    embedding_matrix = np.zeros((vocab_size, vector_size), dtype=np.float32)
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for i, word in enumerate(gensim_words, start=2):
        vocab[word] = i
        embedding_matrix[i] = g_model.wv[word]

    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    print(f"Loaded {vocab_size} vectors with dimension {vector_size}.")
    return vocab, embedding_tensor


def parse_bracket_label(source: str):
    match = BRACKET_RE.search(source.lower())
    if not match:
        return None
    return int(match.group(1)) - 1


class MTGDeckDataset(Dataset):
    def __init__(self, jsonl_filepath: Path, vocab, max_len: int = MAX_LEN):
        self.max_len = max_len
        self.vocab = vocab
        self.inputs = []
        self.labels = []

        print(f"Parsing decks from {jsonl_filepath}...")
        with open(jsonl_filepath, "r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue

                data = json.loads(line)
                label = parse_bracket_label(data.get("source", ""))
                if label is None:
                    continue

                mainboard = list(data.get("mainboard", {}).keys())
                sideboard = list(data.get("sideboard", {}).keys())
                cards = [card for card in (mainboard + sideboard) if card not in BASIC_LANDS]
                if len(cards) < 10:
                    continue

                deck_ids = [self.vocab.get(card, 1) for card in cards[: self.max_len]]
                if len(deck_ids) < self.max_len:
                    deck_ids += [0] * (self.max_len - len(deck_ids))

                self.inputs.append(deck_ids)
                self.labels.append(label)

        if not self.inputs:
            raise ValueError("No labeled decks were found. Check source labels and input file path.")

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Dataset ready: {len(self.inputs)} labeled decks.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class EDHAttentionDeepSets(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        pretrained_weights: torch.Tensor,
        num_heads: int = 4
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(pretrained_weights)

        self.card_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Multi-Head Attention (Pooling by Multihead Attention approach)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        # Learnable query to pool the deck into a single vector
        self.seed_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.deck_mlp = nn.Sequential(
            # Input is *2 because we concatenate Attention output and Max Pool output
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # PyTorch MHA padding mask expects True for elements that should be IGNORED
        key_padding_mask = (x == 0)

        embedded = self.embedding(x)
        card_features = self.card_mlp(embedded) # Shape: (Batch, Seq_Len, Hidden)

        # --- PATH 1: Multi-Head Attention ---
        # Expand our learnable query for the whole batch
        query = self.seed_query.expand(batch_size, -1, -1)
        
        # MHA returns (attn_output, attn_weights)
        attn_out, _ = self.mha(
            query=query, 
            key=card_features, 
            value=card_features, 
            key_padding_mask=key_padding_mask
        )
        attn_pool = attn_out.squeeze(1) # Shape: (Batch, Hidden)

        # --- PATH 2: Max Pooling Bypass ---
        # Mask out padding with a huge negative number before max pooling
        masked_features = card_features.masked_fill(key_padding_mask.unsqueeze(-1), -1e9)
        max_pool_out, _ = torch.max(masked_features, dim=1) # Shape: (Batch, Hidden)

        # --- COMBINE ---
        deck_vector = torch.cat([attn_pool, max_pool_out], dim=1)
        
        return self.deck_mlp(deck_vector)


def evaluate(net: nn.Module, loader: DataLoader, criterion: nn.Module):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def plot_learning_curves(history: dict, trial_name: str, output_path: Path):
    """Generates and saves a matplotlib plot for the train/val curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='orange')
    ax1.set_title(f'Loss: {trial_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy subplot
    ax1.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='orange')
    ax2.set_title(f'Accuracy: {trial_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_one_config(
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    embedding_dim: int,
    pretrained_weights: torch.Tensor,
    trial_name: str
):
    model = EDHAttentionDeepSets(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        pretrained_weights=pretrained_weights,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [
            {"params": model.embedding.parameters(), "lr": config["embedding_lr"]},
            {
                "params": [
                    p for n, p in model.named_parameters() if not n.startswith("embedding.")
                ],
                "lr": config["classifier_lr"],
            },
        ],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_loss)

        # Record metrics for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        print(
            f"  Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

    # Save the plot for this specific trial
    plot_learning_curves(history, trial_name, PLOTS_DIR / f"{trial_name}.png")

    result = {
        "hidden_dim": config["hidden_dim"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "classifier_lr": config["classifier_lr"],
        "embedding_lr": config["embedding_lr"],
        "weight_decay": config["weight_decay"],
        "best_val_acc": best_val_acc,
        "last_val_acc": history['val_acc'][-1],
        "last_val_loss": history['val_loss'][-1],
        "last_train_acc": history['train_acc'][-1],
        "last_train_loss": history['train_loss'][-1],
    }
    return result, best_state_dict


def main() -> None:
    set_seed(SEED)

    vocab, pretrained_weights = load_gensim_embeddings(EMBEDDING_MODEL)
    embedding_dim = pretrained_weights.shape[1]

    dataset = MTGDeckDataset(INPUT_EDH, vocab, max_len=MAX_LEN)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset is too small for a train/validation split.")

    split_gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)

    search_space = []
    for hidden_dim, classifier_lr, dropout, batch_size, weight_decay in itertools.product(
        HIDDEN_DIMS,
        CLASSIFIER_LRS,
        DROPOUTS,
        BATCH_SIZES,
        WEIGHT_DECAYS,
    ):
        search_space.append(
            {
                "hidden_dim": hidden_dim,
                "classifier_lr": classifier_lr,
                "embedding_lr": EMBEDDING_LR,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": EPOCHS_PER_TRIAL,
                "weight_decay": weight_decay,
            }
        )

    print(f"Running {len(search_space)} hyperparameter trials on {DEVICE}...")
    all_results = []
    best_result = None
    best_state_dict = None

    for idx, config in enumerate(search_space, start=1):
        trial_name = f"trial_{idx}_h{config['hidden_dim']}_lr{config['classifier_lr']}_drop{config['dropout']}"
        print(f"\n=== Trial {idx}/{len(search_space)}: {config} ===")

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=DEVICE.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            pin_memory=DEVICE.type == "cuda",
        )

        result, state_dict = train_one_config(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            pretrained_weights=pretrained_weights,
            trial_name=trial_name
        )
        all_results.append(result)

        if best_result is None or result["best_val_acc"] > best_result["best_val_acc"]:
            best_result = result
            best_state_dict = state_dict

    all_results.sort(key=lambda row: row["best_val_acc"], reverse=True)
    RESULTS_OUTPUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    if best_state_dict is not None:
        torch.save(best_state_dict, BEST_WEIGHTS_OUTPUT)

    print("\nTop 5 trials by best validation accuracy:")
    for rank, row in enumerate(all_results[:5], start=1):
        print(
            f"{rank}. val={row['best_val_acc'] * 100:.2f}% | "
            f"hidden={row['hidden_dim']} | lr={row['classifier_lr']} | "
            f"dropout={row['dropout']} | batch={row['batch_size']} | wd={row['weight_decay']}"
        )

    print(f"\nSaved search results to {RESULTS_OUTPUT}")
    print(f"Saved plots to {PLOTS_DIR.absolute()}")
    if best_result is not None:
        print(f"Best val accuracy: {best_result['best_val_acc'] * 100:.2f}%")
        print(f"Saved best weights to {BEST_WEIGHTS_OUTPUT}")


if __name__ == "__main__":
    main()