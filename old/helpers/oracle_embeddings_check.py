import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def normalize_card_name(name: str) -> str:
    """Lowercases card names and keeps the front face for split/DFC names."""
    name = str(name).lower()
    front_face = re.split(r"\s*//\s*", name)[0]
    return front_face.strip()


def load_embedding_bundle(embedding_path: Path) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Loads the saved PyTorch bundle and returns weights + vocab."""
    bundle = torch.load(embedding_path, map_location="cpu")

    if not isinstance(bundle, dict):
        raise ValueError("Expected a dict bundle with keys 'weights' and 'vocab'.")
    if "weights" not in bundle or "vocab" not in bundle:
        raise ValueError("Bundle missing required keys: 'weights' and/or 'vocab'.")

    weights = bundle["weights"]
    vocab = bundle["vocab"]

    if not isinstance(weights, torch.Tensor):
        raise ValueError("'weights' must be a torch.Tensor.")
    if not isinstance(vocab, dict):
        raise ValueError("'vocab' must be a dict[str, int].")

    return weights.float(), vocab


def invert_vocab(vocab: Dict[str, int], size: int) -> List[str]:
    """Builds an id -> token array and validates index bounds."""
    id_to_token = [""] * size
    for token, idx in vocab.items():
        if not isinstance(idx, int):
            raise ValueError(f"Vocab index for '{token}' is not int: {idx}")
        if idx < 0 or idx >= size:
            raise ValueError(f"Vocab index out of bounds for '{token}': {idx}")
        if id_to_token[idx]:
            raise ValueError(f"Duplicate vocab id {idx} for '{id_to_token[idx]}' and '{token}'")
        id_to_token[idx] = token

    missing = [i for i, t in enumerate(id_to_token) if not t]
    if missing:
        raise ValueError(f"Missing token assignments for ids: {missing[:10]}")

    return id_to_token


def cosine_neighbors(weights: torch.Tensor, query_idx: int, topn: int, ban_ids: List[int]) -> List[Tuple[int, float]]:
    """Returns top cosine neighbors for a row vector."""
    vec = weights[query_idx]
    vec_norm = torch.linalg.norm(vec)
    if vec_norm == 0:
        return []

    all_norms = torch.linalg.norm(weights, dim=1)
    denom = all_norms * vec_norm

    scores = torch.zeros(weights.shape[0], dtype=torch.float32)
    valid = denom > 0
    if valid.any():
        scores[valid] = (weights[valid] @ vec) / denom[valid]

    for idx in ban_ids:
        if 0 <= idx < scores.shape[0]:
            scores[idx] = -math.inf

    top_values, top_indices = torch.topk(scores, k=min(topn, scores.shape[0]))
    return [(int(i), float(v)) for i, v in zip(top_indices, top_values) if math.isfinite(float(v))]


def summarize_norms(weights: torch.Tensor, w2v_dim: int) -> None:
    """Prints basic norm diagnostics for each segment."""
    core = weights[2:] if weights.shape[0] > 2 else weights
    if core.numel() == 0:
        print("No non-special-token rows found for norm summary.")
        return

    w2v = core[:, :w2v_dim]
    nlp = core[:, w2v_dim:]

    w2v_norms = torch.linalg.norm(w2v, dim=1)
    nlp_norms = torch.linalg.norm(nlp, dim=1)

    print("\nNorm summary (excluding <PAD>/<UNK>):")
    print(
        "  W2V norms -> "
        f"min={w2v_norms.min().item():.4f}, "
        f"mean={w2v_norms.mean().item():.4f}, "
        f"max={w2v_norms.max().item():.4f}"
    )
    print(
        "  NLP norms -> "
        f"min={nlp_norms.min().item():.4f}, "
        f"mean={nlp_norms.mean().item():.4f}, "
        f"max={nlp_norms.max().item():.4f}"
    )

    zero_nlp = int((nlp_norms == 0).sum().item())
    print(f"  Zero NLP vectors: {zero_nlp}/{nlp_norms.numel()}")


def run_checks(embedding_path: Path, w2v_dim: int, topn: int, sample_cards: List[str]) -> None:
    print(f"Loading embedding bundle: {embedding_path}")
    weights, vocab = load_embedding_bundle(embedding_path)

    rows, cols = weights.shape
    print(f"Weights shape: ({rows}, {cols})")
    print(f"Vocab size: {len(vocab)}")

    if rows != len(vocab):
        raise ValueError(f"Row count ({rows}) does not match vocab size ({len(vocab)}).")
    if cols <= w2v_dim:
        raise ValueError(f"Total dim ({cols}) must be > w2v_dim ({w2v_dim}).")

    if "<PAD>" not in vocab or "<UNK>" not in vocab:
        raise ValueError("Vocab missing required special tokens: <PAD> and/or <UNK>.")
    if vocab["<PAD>"] != 0 or vocab["<UNK>"] != 1:
        raise ValueError("Expected <PAD>=0 and <UNK>=1.")

    if torch.isnan(weights).any() or torch.isinf(weights).any():
        raise ValueError("Found NaN or Inf in embedding weights.")

    id_to_token = invert_vocab(vocab, rows)

    print("Basic checks: PASS")
    print(f"Derived NLP dim: {cols - w2v_dim}")
    summarize_norms(weights, w2v_dim)

    print("\nSample similarity checks:")
    for raw_card in sample_cards:
        card = normalize_card_name(raw_card)
        idx = vocab.get(card)
        if idx is None:
            print(f"  - {raw_card} -> '{card}' not in vocab")
            continue

        neighbors = cosine_neighbors(weights, idx, topn=topn + 3, ban_ids=[0, 1, idx])
        print(f"  - {raw_card} -> '{card}' (id={idx})")
        for n_idx, score in neighbors[:topn]:
            print(f"      {id_to_token[n_idx]} (cos={score:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inspect hybrid oracle embeddings built by generate_oracle_embeddings.py"
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("../models/embeddings/embedding-models/896dim_oracle_embeddings.pt"),
        help="Path to saved embedding bundle (.pt).",
    )
    parser.add_argument(
        "--w2v-dim",
        type=int,
        default=512,
        help="Word2Vec dimensionality used before concatenating NLP vectors.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=5,
        help="How many nearest neighbors to show per sample card.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=[
            "Ponder",
            "Sol Ring",
            "Lightning Bolt",
            "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki",
            "Demonic Tutor",
            "Mystical Tutor",
            "The One Ring",
            "Dark Ritual",
            "Island",
            "Snow-Covered Island"
        ],
        help="Sample cards for nearest-neighbor sanity checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_checks(
        embedding_path=args.embedding_path,
        w2v_dim=args.w2v_dim,
        topn=args.topn,
        sample_cards=args.samples,
    )
