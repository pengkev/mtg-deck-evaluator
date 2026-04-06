#!/usr/bin/env python3
"""Streamlit UI for scoring EDH decklists with the SetTransformer model."""

from __future__ import annotations

import os
import re
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

SAMPLE_MAINBOARD = """
1 Ajani, Sleeper Agent
1 Arcane Signet
1 Astral Cornucopia
1 Birds of Paradise
1 Blightbelly Rat
1 Blighted Agent
1 Bloated Contaminator
1 Breeding Pool
1 Brokers Ascendancy
1 Cankerbloom
1 Chromatic Lantern
1 Command Tower
1 Contagion Engine
1 Contaminant Grafter
1 Counterspell
1 Cultivate
1 Cyclonic Rift
1 Deepglow Skate
1 Doubling Season
1 Dreamtide Whale
1 Drown in Ichor
1 Everflowing Chalice
1 Evolution Sage
1 Exotic Orchard
1 Experimental Augury
1 Ezuri, Stalker of Spheres
1 Farseek
1 Fellwar Stone
1 Flooded Strand
1 Flux Channeler
4 Forest
1 Glistening Sphere
1 Godless Shrine
1 Hallowed Fountain
1 Ichor Rats
1 Indatha Triome
1 Inexorable Tide
1 Infectious Bite
1 Infectious Inquiry
1 Innkeeper's Talent
3 Island
1 Ixhel, Scion of Atraxa
1 Karn's Bastion
1 Lightning Greaves
1 Marsh Flats
1 Metastatic Evangel
1 Misty Rainforest
1 Narset, Parter of Veils
1 Nature's Lore
1 Norn's Decree
1 Oko, Thief of Crowns
1 Opulent Palace
1 Overgrown Tomb
1 Path of Ancestry
1 Path to Exile
1 Phyresis Outbreak
1 Phyrexian Swarmlord
1 Plague Stinger
3 Plains
1 Polluted Delta
1 Prologue to Phyresis
1 Raffine's Tower
1 Rhystic Study
1 Sandsteppe Citadel
1 Seaside Citadel
1 Skithiryx, the Blight Dragon
1 Skrelv, Defector Mite
1 Smothering Tithe
1 Sol Ring
1 Spara's Headquarters
3 Swamp
1 Swords to Plowshares
1 Tainted Observer
1 Tamiyo, Field Researcher
1 Teferi's Protection
1 Teferi, Master of Time
1 Tekuthal, Inquiry Dominus
1 Temple Garden
1 Tezzeret's Gambit
1 Thrummingbird
1 Unnatural Restoration
1 Venerated Rotpriest
1 Verdant Catacombs
1 Voidwing Hybrid
1 Vorinclex, Monstrous Raider
1 Vraska's Fall
1 Vraska, Betrayal's Sting
1 Watery Grave
1 Windswept Heath
1 Zagoth Triome
"""

SAMPLE_COMPANION = ""  # Optional


def parse_card_lines(text: str) -> dict[str, int]:
    cards: dict[str, int] = {}
    qty_name_re = re.compile(r"^(\d+)\s*x?\s+(.+?)$")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = qty_name_re.match(line)
        if m:
            qty = int(m.group(1))
            name = m.group(2).strip()
        else:
            qty = 1
            name = line
        if not name:
            continue
        qty = max(1, int(qty))
        cards[name] = cards.get(name, 0) + qty
    return cards


def _single_card_name(cards: dict[str, int]) -> Optional[str]:
    if not cards:
        return None
    if len(cards) != 1:
        return None
    (name, qty), *_ = cards.items()
    if int(qty) != 1:
        return None
    return str(name)


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


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    st.set_page_config(page_title="MTG Deck Evaluator", page_icon="cards", layout="wide")
    st.title("MTG Deck Evaluator")
    st.caption("Score EDH decklists with the trained SetTransformer model.")

    # Hide model settings in deployed app by default.
    # To show them locally/dev, set env var `MTG_SHOW_MODEL_SETTINGS=1` or Streamlit secret `show_model_settings=true`.
    show_model_settings = _truthy(os.getenv("MTG_SHOW_MODEL_SETTINGS")) or _truthy(
        getattr(st, "secrets", {}).get("show_model_settings", False)
    )

    checkpoint = str(DEFAULT_CHECKPOINT)
    embeddings = str(DEFAULT_EMBEDDINGS)
    calibrator_raw = str(DEFAULT_CALIBRATOR)
    device_option = "Auto"
    max_deck_len = MAX_DECK_LEN
    max_qty_embed = MAX_QTY_EMBED

    if show_model_settings:
        with st.sidebar:
            st.header("Model Settings")
            checkpoint = st.text_input("Checkpoint", value=checkpoint)
            embeddings = st.text_input("Embeddings", value=embeddings)
            calibrator_raw = st.text_input("Calibrator (optional)", value=calibrator_raw)

            device_option = st.selectbox("Device", options=["Auto", "CPU", "CUDA"], index=0)
            max_deck_len = st.number_input(
                "Max deck length",
                min_value=50,
                max_value=200,
                value=int(max_deck_len),
                step=1,
            )
            max_qty_embed = st.number_input(
                "Max quantity embedding",
                min_value=1,
                max_value=200,
                value=int(max_qty_embed),
                step=1,
            )

    left, right = st.columns([2, 1])
    with left:
        commander_text = st.text_area(
            "Commander Cards",
            value=SAMPLE_COMMANDERS,
            height=120,
            help="Put only commander card lines here (for example: '1 Atraxa, Praetors\' Voice').",
        )
        companion_text = st.text_area(
            "Companion (optional)",
            value=SAMPLE_COMPANION,
            height=90,
            help="Optional single companion card (one line, qty 1). This will be added into the model input behind the scenes.",
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

    commander_cards = parse_card_lines(commander_text)
    main_cards = parse_card_lines(mainboard_text)
    companion_cards = parse_card_lines(companion_text)

    num_commander_cards = int(sum(commander_cards.values()))
    if num_commander_cards not in {1, 2}:
        st.error("Commander section must contain exactly 1 card (or 2 if using Partner commanders).")
        return
    if any(int(qty) != 1 for qty in commander_cards.values()):
        st.error("Commander entries must each have quantity 1.")
        return

    expected_main = 100 - num_commander_cards
    actual_main = int(sum(main_cards.values()))
    if actual_main != expected_main:
        st.error(f"Main deck must contain exactly {expected_main} cards for this commander setup (currently {actual_main}).")
        return

    companion_name = None
    if companion_cards:
        companion_name = _single_card_name(companion_cards)
        if companion_name is None:
            st.error("Companion must be a single card line with quantity 1.")
            return
        if companion_name in main_cards:
            st.error("Companion card should not also be listed in the main deck.")
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

        deck_obj = {
            "cmds": commander_cards,
            "cards": dict(main_cards),
        }
        if companion_name is not None:
            deck_obj["cards"][companion_name] = deck_obj["cards"].get(companion_name, 0) + 1

        status = st.status("Vectorizing deck...", expanded=True)
        progress = st.progress(0, text="Vectorizing cards")
        last_card = st.empty()

        def on_progress(stage: str, i: int, total: int, message: str) -> None:
            if stage == "vectorize":
                pct = 0.0 if total <= 0 else float(i) / float(total)
                progress.progress(min(1.0, max(0.0, pct)), text=f"Vectorizing cards ({i}/{total})")
                last_card.write(f"Last: {message}")
                status.update(label="Vectorizing deck...", state="running")
            elif stage == "model":
                status.update(label="Running model...", state="running")
            elif stage == "calibrate":
                status.update(label="Calibrating score...", state="running")

        scores = pipeline.score_deck_obj(deck_obj, progress_callback=on_progress)
        status.update(label="Done", state="complete")
    except Exception as exc:
        st.exception(exc)
        return

    st.metric("Calibrated Score", f"{scores['calibrated_score']:.3f}")


if __name__ == "__main__":
    main()
