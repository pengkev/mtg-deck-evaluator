import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "microsoft/MiniLM-L12-H384-uncased"
DEFAULT_ORACLE_JSON = Path("../../data/oracle_cards.json")
DEFAULT_OUTPUT_DIR = Path("embedding-models/mtg-minilm-mlm")


def normalize_card_name(name: str) -> str:
    """Normalizes names consistently with embedding generation."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"^a-", "", name)
    name = re.sub(r"\s+", " ", name)
    front_face = re.split(r"\s*//\s*", name)[0]
    return front_face.strip()


def build_card_metadata_header(card: Dict[str, Any]) -> str:
    """Builds a semantic description of the card's physical properties."""
    chunks: List[str] = []

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
    parts: List[str] = []

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
    deduped = list(dict.fromkeys(parts))
    return "\n".join(deduped)


def build_oracle_text_corpus(oracle_json_path: Path) -> List[str]:
    """Builds de-duplicated training corpus from Oracle cards."""
    with open(oracle_json_path, "r", encoding="utf-8") as f:
        cards_data = json.load(f)

    # Use normalized-name dict so we keep one consolidated text per card identity.
    oracle_index: Dict[str, str] = {}
    for card in cards_data:
        candidate_names = [card.get("name", "")]
        candidate_names.extend(face.get("name", "") for face in card.get("card_faces", []))

        normalized_names = {normalize_card_name(name) for name in candidate_names if name}
        normalized_names = {name for name in normalized_names if name}
        if not normalized_names:
            continue

        combined_text = get_combined_oracle_text(card)
        for norm_name in normalized_names:
            existing_text = oracle_index.get(norm_name, "")
            if not existing_text.strip() and combined_text.strip():
                oracle_index[norm_name] = combined_text
            elif norm_name not in oracle_index:
                oracle_index[norm_name] = combined_text

    corpus = [text for text in oracle_index.values() if text.strip()]
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM with MLM on MTG Oracle text")
    parser.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Loading Oracle corpus from {args.oracle_json}...")
    texts = build_oracle_text_corpus(args.oracle_json)
    if not texts:
        raise ValueError("No oracle text found for MLM training.")
    print(f"Corpus size: {len(texts)}")

    dataset = Dataset.from_dict({"text": texts})

    print(f"Loading tokenizer/model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.max_length) * args.max_length
        result = {
            k: [t[i : i + args.max_length] for i in range(0, total_length, args.max_length)]
            for k, t in concatenated.items()
        }
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    print("Starting MLM fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("Done. generate_oracle_embeddings.py will now auto-load this checkpoint if present.")


if __name__ == "__main__":
    main()
