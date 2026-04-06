#!/usr/bin/env python3
"""Score EDH decklists with the trained SetTransformer checkpoint.

This script mirrors the notebook inference pipeline:
1) Load hybrid embeddings (vocab + static weights)
2) Build SetTransformer with checkpoint config
3) Load model weights
4) Optionally load isotonic calibrator
5) Parse plain-text decklists and output raw / bounded / calibrated scores
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"
MAX_DECK_LEN = 115
MAX_QTY_EMBED = 50


def normalize_card_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"^a-", "", name)
    name = re.sub(r"\s+", " ", name)
    front_face = re.split(r"\s*//\s*", name)[0]
    return front_face.strip()


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


def extract_card_qty_role_triplets(deck_obj: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    triplets: List[Tuple[str, int, int]] = []

    for cmd_name, cmd_qty in _parse_name_qty_items(deck_obj.get("cmds", [])):
        triplets.append((cmd_name, cmd_qty, 1))

    for zone in ("main", "mainboard", "sideboard"):
        for name, qty in _parse_name_qty_items(deck_obj.get(zone, [])):
            triplets.append((name, qty, 0))

    for name, qty in _parse_name_qty_items(deck_obj.get("cards", {})):
        triplets.append((name, qty, 0))

    collapsed: Dict[Tuple[str, int], int] = {}
    for card_name, qty, role_id in triplets:
        normalized = normalize_card_name(card_name)
        qty_int = int(qty)
        if not normalized or qty_int <= 0:
            continue
        key = (normalized, int(role_id))
        collapsed[key] = collapsed.get(key, 0) + qty_int

    return [(name, qty, role_id) for (name, role_id), qty in collapsed.items()]


def parse_plaintext_decklist(decklist_text: str) -> Dict[str, Any]:
    """Parse common decklist text into notebook-compatible deck object.

    Supported formats per line:
    - "1 Sol Ring"
    - "1x Sol Ring"
    - "Sol Ring" (defaults to qty=1)

    Section headers recognized:
    - "Commander" / "Commanders"
    """
    cards: Dict[str, int] = {}
    commanders: Dict[str, int] = {}

    in_commander = False

    qty_name_re = re.compile(r"^(\d+)\s*x?\s+(.+?)$")
    for raw_line in decklist_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower().rstrip(":")
        if lower in {"commander", "commanders"}:
            in_commander = True
            continue

        m = qty_name_re.match(line)
        if m:
            qty = int(m.group(1))
            name = m.group(2).strip()
        else:
            # Allow no-qty commander lines like "Stella Lee, Wild Card"
            qty = 1
            name = line

        if not name:
            continue

        target = commanders if in_commander else cards
        target[name] = target.get(name, 0) + max(1, qty)

    return {"cards": cards, "cmds": commanders}


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
        pad_token_id: int,
        qty_embed_dim: int = 32,
        role_embed_dim: int = 8,
        dropout: float = 0.1,
        pretrained_weights: Optional[torch.Tensor] = None,
        land_bin_classes: int = 5,
        ramp_bin_classes: int = 5,
    ):
        super().__init__()

        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=pad_token_id)
        if pretrained_weights is not None:
            self.card_embedding.weight.data.copy_(pretrained_weights)

        self.qty_embedding = nn.Embedding(max_qty_embed + 1, qty_embed_dim, padding_idx=0)
        self.role_embedding = nn.Embedding(3, role_embed_dim, padding_idx=2)

        self.input_proj = nn.Sequential(
            nn.Linear(card_embed_dim + qty_embed_dim + role_embed_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        self.pma_seeds = nn.Parameter(torch.randn(num_pma_seeds, model_dim))
        self.pma_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        pooled_dim = model_dim * num_pma_seeds
        self.output_layer = nn.Linear(pooled_dim, 1)
        self.mlm_head = nn.Linear(model_dim, vocab_size)
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
        return self.transformer_encoder(x, src_key_padding_mask=~mask)

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

    def forward_raw(self, card_ids: torch.Tensor, qty_ids: torch.Tensor, role_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


@dataclass
class LoadedPipeline:
    model: SetTransformer
    calibrator: Any
    vocab: Dict[str, int]
    pad_token_id: int
    unk_token_id: int
    max_qty_embed: int
    max_deck_len: int
    device: torch.device

    def prepare_deck(self, deck_obj: Dict[str, Any]) -> "PreparedDeck":
        triplets = extract_card_qty_role_triplets(deck_obj)
        if not triplets:
            raise ValueError("Deck has no valid cards after parsing/normalization")

        selected_triplets = triplets[: self.max_deck_len]
        truncated_triplets = max(0, len(triplets) - len(selected_triplets))

        card_ids: List[int] = []
        qty_ids: List[int] = []
        role_ids: List[int] = []
        unknown_cards: List[str] = []

        for card_name, qty, role in selected_triplets:
            card_id = self.vocab.get(card_name, self.unk_token_id)
            card_ids.append(card_id)
            qty_ids.append(max(1, min(int(qty), self.max_qty_embed)))
            role_ids.append(int(role))
            if card_id == self.unk_token_id:
                unknown_cards.append(card_name)

        mask = [1] * len(card_ids)
        pad_len = self.max_deck_len - len(card_ids)
        if pad_len > 0:
            card_ids += [self.pad_token_id] * pad_len
            qty_ids += [0] * pad_len
            role_ids += [2] * pad_len
            mask += [0] * pad_len

        return PreparedDeck(
            card_ids=card_ids,
            qty_ids=qty_ids,
            role_ids=role_ids,
            mask=mask,
            unique_normalized_cards=len(selected_triplets),
            unknown_cards=sorted(set(unknown_cards)),
            num_commander_cards=int(sum(deck_obj.get("cmds", {}).values())),
            num_main_cards=int(sum(deck_obj.get("cards", {}).values())),
            truncated_triplets=truncated_triplets,
        )

    @torch.no_grad()
    def score_deck_obj(self, deck_obj: Dict[str, Any]) -> Dict[str, float]:
        prepared = self.prepare_deck(deck_obj)

        batch_card_ids = torch.tensor([prepared.card_ids], dtype=torch.long, device=self.device)
        batch_qty_ids = torch.tensor([prepared.qty_ids], dtype=torch.long, device=self.device)
        batch_role_ids = torch.tensor([prepared.role_ids], dtype=torch.long, device=self.device)
        batch_mask = torch.tensor([prepared.mask], dtype=torch.bool, device=self.device)

        raw_score = float(self.model.forward_raw(batch_card_ids, batch_qty_ids, batch_role_ids, batch_mask).item())
        bounded_score = float(self.model(batch_card_ids, batch_qty_ids, batch_role_ids, batch_mask).item())

        if self.calibrator is not None:
            calibrated_score = float(self.calibrator.predict([raw_score])[0])
        else:
            calibrated_score = bounded_score

        return {
            "raw_score": raw_score,
            "bounded_score": bounded_score,
            "calibrated_score": calibrated_score,
        }


@dataclass
class PreparedDeck:
    card_ids: List[int]
    qty_ids: List[int]
    role_ids: List[int]
    mask: List[int]
    unique_normalized_cards: int
    unknown_cards: List[str]
    num_commander_cards: int
    num_main_cards: int
    truncated_triplets: int


def load_pipeline(
    checkpoint_path: Path,
    embedding_path: Path,
    calibrator_path: Optional[Path],
    device: torch.device,
    max_deck_len: int,
    max_qty_embed: int,
) -> LoadedPipeline:
    blob = torch.load(embedding_path, map_location="cpu")
    vocab = dict(blob["vocab"])
    weights = blob["weights"].float()

    if MASK_TOKEN not in vocab:
        vocab[MASK_TOKEN] = weights.shape[0]
        weights = torch.cat([weights, torch.zeros(1, weights.shape[1], dtype=weights.dtype)], dim=0)

    pad_token_id = int(vocab[PAD_TOKEN])
    unk_token_id = int(vocab[UNK_TOKEN])

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = dict(ckpt.get("config", {}))
    else:
        state_dict = ckpt
        config = {}

    land_bin_edges = config.get("phase1_land_bin_edges", [30, 34, 37, 40])
    ramp_bin_edges = config.get("phase1_ramp_bin_edges", [6, 10, 14, 18])

    model = SetTransformer(
        vocab_size=len(vocab),
        card_embed_dim=int(weights.shape[1]),
        model_dim=int(config.get("model_dim", 512)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        num_heads=int(config.get("num_heads", 8)),
        num_blocks=int(config.get("num_blocks", 4)),
        num_pma_seeds=int(config.get("num_pma_seeds", 4)),
        max_qty_embed=int(config.get("max_qty_embed", max_qty_embed)),
        pad_token_id=pad_token_id,
        qty_embed_dim=int(config.get("qty_embed_dim", 32)),
        role_embed_dim=int(config.get("role_embed_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        pretrained_weights=weights,
        land_bin_classes=len(land_bin_edges) + 1,
        ramp_bin_classes=len(ramp_bin_edges) + 1,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    calibrator = None
    if calibrator_path is not None and calibrator_path.exists():
        if joblib is None:
            raise ImportError("joblib is required to load the score calibrator")
        calibrator = joblib.load(calibrator_path)

    return LoadedPipeline(
        model=model,
        calibrator=calibrator,
        vocab=vocab,
        pad_token_id=pad_token_id,
        unk_token_id=unk_token_id,
        max_qty_embed=int(config.get("max_qty_embed", max_qty_embed)),
        max_deck_len=max_deck_len,
        device=device,
    )


def read_decklist_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Score a decklist with the trained SetTransformer pipeline")
    parser.add_argument("--decklist-file", type=Path, help="Path to plaintext decklist")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=here / "checkpoints" / "set_transformer_master_run.pt",
        help="Trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=here.parent / "embeddings" / "embedding-models" / "896dim_oracle_embeddings.pt",
        help="Embedding tensor file with vocab + weights",
    )
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=here / "checkpoints" / "set_transformer_master_run_isotonic_calibrator.joblib",
        help="Isotonic calibrator .joblib path (optional)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-deck-len", type=int, default=MAX_DECK_LEN)
    parser.add_argument("--max-qty-embed", type=int, default=MAX_QTY_EMBED)
    parser.add_argument("--json", action="store_true", help="Print only JSON output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    decklist_text = ""
    if args.decklist_file is not None:
        decklist_text = read_decklist_text(args.decklist_file)
    else:
        # Support piping from stdin
        import sys

        if not sys.stdin.isatty():
            decklist_text = sys.stdin.read()

    if not decklist_text.strip():
        raise SystemExit("No decklist provided. Pass --decklist-file or pipe deck text via stdin.")

    deck_obj = parse_plaintext_decklist(decklist_text)

    calibrator_path = args.calibrator
    if calibrator_path is not None and not calibrator_path.exists():
        calibrator_path = None

    pipeline = load_pipeline(
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding_path,
        calibrator_path=calibrator_path,
        device=torch.device(args.device),
        max_deck_len=int(args.max_deck_len),
        max_qty_embed=int(args.max_qty_embed),
    )

    scores = pipeline.score_deck_obj(deck_obj)

    out = {
        "checkpoint": str(args.checkpoint),
        "embedding_path": str(args.embedding_path),
        "calibrator": str(calibrator_path) if calibrator_path is not None else None,
        "num_main_cards": int(sum(deck_obj.get("cards", {}).values())),
        "num_commanders": int(sum(deck_obj.get("cmds", {}).values())),
        **scores,
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("Scoring result")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
