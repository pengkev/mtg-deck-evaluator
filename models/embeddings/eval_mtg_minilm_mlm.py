import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from finetune_mtg_minilm_mlm import BASE_MODEL, DEFAULT_ORACLE_JSON, DEFAULT_OUTPUT_DIR, build_oracle_text_corpus


def tokenize_and_chunk(texts: List[str], tokenizer, max_length: int) -> Dataset:
    """Tokenizes texts and groups tokens into fixed-size MLM blocks."""
    raw_ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // max_length) * max_length
        if total_length == 0:
            return {k: [] for k in concatenated.keys()}
        return {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated.items()
        }

    return tokenized.map(group_texts, batched=True)


def evaluate_model(
    model_name_or_path: str,
    eval_dataset: Dataset,
    tokenizer,
    batch_size: int,
    mlm_probability: float,
    seed: int,
) -> Dict[str, float]:
    """Evaluates masked-token loss/perplexity for a given model."""
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    args = TrainingArguments(
        output_dir="./.tmp_eval",
        per_device_eval_batch_size=batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    perplexity = float(math.exp(eval_loss)) if eval_loss < 20 else float("inf")

    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick MLM eval for MTG MiniLM on held-out Oracle text")
    parser.add_argument("--oracle-json", type=Path, default=DEFAULT_ORACLE_JSON)
    parser.add_argument("--finetuned-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-base", action="store_true", help="Also evaluate the base model for comparison")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading Oracle corpus from {args.oracle_json}...")
    texts = build_oracle_text_corpus(args.oracle_json)
    texts = [t for t in texts if t.strip()]
    if len(texts) < 100:
        raise ValueError("Oracle corpus is too small for a useful held-out evaluation.")

    full_ds = Dataset.from_dict({"text": texts})
    split = full_ds.train_test_split(test_size=args.eval_split, seed=args.seed)
    eval_texts = split["test"]["text"]

    print(f"Total documents: {len(texts)}")
    print(f"Eval documents: {len(eval_texts)}")

    tokenizer_source = str(args.finetuned_dir) if args.finetuned_dir.exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    eval_dataset = tokenize_and_chunk(eval_texts, tokenizer, args.max_length)

    if len(eval_dataset) == 0:
        raise ValueError("Eval token chunks are empty. Lower --max-length or increase corpus size.")

    print(f"Eval chunks: {len(eval_dataset)}")

    if not args.finetuned_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {args.finetuned_dir}. Run finetune_mtg_minilm_mlm.py first."
        )

    print(f"Evaluating fine-tuned model: {args.finetuned_dir}")
    tuned_metrics = evaluate_model(
        model_name_or_path=str(args.finetuned_dir),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
    )

    print("\nFine-tuned model metrics")
    print(f"  eval_loss:   {tuned_metrics['eval_loss']:.4f}")
    print(f"  perplexity:  {tuned_metrics['perplexity']:.4f}")

    if args.compare_base:
        print(f"\nEvaluating base model: {args.base_model}")
        base_metrics = evaluate_model(
            model_name_or_path=args.base_model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            mlm_probability=args.mlm_probability,
            seed=args.seed,
        )

        print("\nBase model metrics")
        print(f"  eval_loss:   {base_metrics['eval_loss']:.4f}")
        print(f"  perplexity:  {base_metrics['perplexity']:.4f}")

        loss_delta = base_metrics["eval_loss"] - tuned_metrics["eval_loss"]
        ppl_delta = base_metrics["perplexity"] - tuned_metrics["perplexity"]

        print("\nDelta (base - fine-tuned)")
        print(f"  eval_loss delta:  {loss_delta:+.4f}")
        print(f"  perplexity delta: {ppl_delta:+.4f}")


if __name__ == "__main__":
    main()
