#!/usr/bin/env python3
"""Streamlit UI for scoring EDH decklists with the SetTransformer model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import torch

from score_decklist import (
    MAX_DECK_LEN,
    MAX_QTY_EMBED,
    load_pipeline,
    parse_plaintext_decklist,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "set_transformer_master_run.pt"
DEFAULT_EMBEDDINGS = HERE / "896dim_oracle_embeddings.pt"
DEFAULT_CALIBRATOR = HERE / "set_transformer_master_run_isotonic_calibrator.joblib"

SAMPLE_COMMANDERS = """1 Atraxa, Praetors' Voice"""

SAMPLE_MAINBOARD = """1 Sol Ring
1 Arcane Signet
1 Fellwar Stone
1 Rhystic Study
1 Smothering Tithe
1 Cyclonic Rift
1 Swords to Plowshares
1 Demonic Tutor
1 Vampiric Tutor
1 Mana Crypt
1 Enlightened Tutor
1 Swan Song
1 Force of Will
1 Nature's Claim
1 Esper Sentinel
1 Mystic Remora
"""


@st.cache_resource(show_spinner=False)
def get_pipeline(
    checkpoint_path: str,
    embedding_path: str,
    calibrator_path: Optional[str],
    device_name: str,
    max_deck_len: int,
    max_qty_embed: int,
):
    calibrator = Path(calibrator_path) if calibrator_path else None
    return load_pipeline(
        checkpoint_path=Path(checkpoint_path),
        embedding_path=Path(embedding_path),
        calibrator_path=calibrator,
        device=torch.device(device_name),
        max_deck_len=max_deck_len,
        max_qty_embed=max_qty_embed,
    )


def pick_device(option: str) -> str:
    if option == "Auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if option == "CUDA" and not torch.cuda.is_available():
        return "cpu"
    return option.lower()


def build_deck_input(commander_text: str, mainboard_text: str) -> str:
    commander_lines = [line.strip() for line in commander_text.splitlines() if line.strip()]
    mainboard_lines = [line.strip() for line in mainboard_text.splitlines() if line.strip()]

    sections = []
    if commander_lines:
        sections.append("Commander")
        sections.extend(commander_lines)
    if mainboard_lines:
        if sections:
            sections.append("")
        sections.append("Mainboard")
        sections.extend(mainboard_lines)
    return "\n".join(sections)


def main() -> None:
    st.set_page_config(page_title="MTG Deck Evaluator", page_icon="cards", layout="wide")
    st.title("MTG Deck Evaluator")
    st.caption("Score EDH decklists with the trained SetTransformer model.")

    with st.sidebar:
        st.header("Model Settings")
        checkpoint = st.text_input("Checkpoint", value=str(DEFAULT_CHECKPOINT))
        embeddings = st.text_input("Embeddings", value=str(DEFAULT_EMBEDDINGS))
        calibrator_raw = st.text_input("Calibrator (optional)", value=str(DEFAULT_CALIBRATOR))

        device_option = st.selectbox("Device", options=["Auto", "CPU", "CUDA"], index=0)
        max_deck_len = st.number_input("Max deck length", min_value=50, max_value=200, value=MAX_DECK_LEN, step=1)
        max_qty_embed = st.number_input("Max quantity embedding", min_value=1, max_value=200, value=MAX_QTY_EMBED, step=1)

    left, right = st.columns([2, 1])
    with left:
        commander_text = st.text_area(
            "Commander Cards",
            value=SAMPLE_COMMANDERS,
            height=120,
            help="Put only commander card lines here (for example: '1 Atraxa, Praetors\' Voice').",
        )
        mainboard_text = st.text_area(
            "Main Deck Cards",
            value=SAMPLE_MAINBOARD,
            height=300,
            help="Use lines like '1 Sol Ring' or 'Card Name' (defaults to qty 1).",
        )
    with right:
        st.markdown("### Input Format")
        st.write("- `1 Card Name`")
        st.write("- `1x Card Name`")
        st.write("- `Card Name` (defaults to qty 1)")
        st.write("- Commander entries go in the Commander Cards box")

    score_clicked = st.button("Score Deck", type="primary", use_container_width=True)

    if not score_clicked:
        return

    deck_text = build_deck_input(commander_text, mainboard_text)

    if not deck_text.strip():
        st.error("Provide a decklist before scoring.")
        return

    resolved_device = pick_device(device_option)
    if device_option == "CUDA" and resolved_device == "cpu":
        st.warning("CUDA was selected, but no CUDA device is available. Falling back to CPU.")

    checkpoint_path = Path(checkpoint)
    embedding_path = Path(embeddings)
    calibrator_path = Path(calibrator_raw) if calibrator_raw.strip() else None
    if calibrator_path is not None and not calibrator_path.exists():
        calibrator_path = None

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return
    if not embedding_path.exists():
        st.error(f"Embeddings file not found: {embedding_path}")
        return

    try:
        with st.spinner("Loading model pipeline..."):
            pipeline = get_pipeline(
                checkpoint_path=str(checkpoint_path),
                embedding_path=str(embedding_path),
                calibrator_path=str(calibrator_path) if calibrator_path is not None else None,
                device_name=resolved_device,
                max_deck_len=int(max_deck_len),
                max_qty_embed=int(max_qty_embed),
            )

        deck_obj = parse_plaintext_decklist(deck_text)
        scores = pipeline.score_deck_obj(deck_obj)
    except Exception as exc:
        st.exception(exc)
        return

    st.metric("Calibrated Score", f"{scores['calibrated_score']:.3f}")


if __name__ == "__main__":
    main()
