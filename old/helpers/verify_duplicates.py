"""
verify_duplicates.py
--------------------
Checks JSONL deck datasets for duplicate deck IDs and duplicate deck content.

Usage:
    python verify_duplicates.py --file ../data/edh-decks.jsonl
    python verify_duplicates.py --file ../data/edh-decks.jsonl --dedupe
    python verify_duplicates.py --root ../data --max-examples 10
    python verify_duplicates.py --no-fail
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_PATTERNS = ["*.jsonl", "*.jsonl.gz"]


def open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def normalize_count(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def normalize_board(board):
    if board is None:
        return []
    if isinstance(board, dict):
        items = [(str(name), normalize_count(count)) for name, count in board.items()]
        return sorted(items)
    if isinstance(board, list):
        counts = Counter(str(card) for card in board)
        return sorted(counts.items())
    return [("raw", str(board))]


def deck_signature(deck: dict) -> str:
    signature = {}
    for key in ("mainboard", "sideboard", "commanders", "commander", "companion"):
        if key in deck:
            signature[key] = normalize_board(deck.get(key))
    if not signature:
        signature = deck
    sig_str = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(sig_str.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[int, str, dict | None, bool]]:
    with open_jsonl(path) as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_line = line.rstrip("\n")
            try:
                yield line_num, raw_line, json.loads(raw_line), False
            except json.JSONDecodeError:
                yield line_num, raw_line, None, True


def check_file(path: Path, max_examples: int, output_path: Path | None = None) -> dict:
    seen_ids = {}
    seen_sigs = {}

    dup_id_total = 0
    dup_id_unique = set()
    dup_sig_total = 0
    dup_sig_unique = set()

    dup_id_examples = []
    dup_sig_examples = []

    total = 0
    parse_errors = 0
    written = 0

    out_handle = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = open_output(output_path)

    try:
        for line_num, raw_line, deck, parse_error in iter_jsonl(path):
            total += 1
            if parse_error:
                parse_errors += 1
                if out_handle:
                    out_handle.write(raw_line + "\n")
                    written += 1
                continue

            deck_id = deck.get("deck_id")
            is_dup_id = False
            if deck_id is not None:
                deck_id = str(deck_id)
                if deck_id in seen_ids:
                    is_dup_id = True
                    dup_id_total += 1
                    dup_id_unique.add(deck_id)
                    if len(dup_id_examples) < max_examples:
                        dup_id_examples.append((deck_id, seen_ids[deck_id], line_num))
                else:
                    seen_ids[deck_id] = line_num

            sig = deck_signature(deck)
            is_dup_sig = False
            if sig in seen_sigs:
                is_dup_sig = True
                dup_sig_total += 1
                dup_sig_unique.add(sig)
                if len(dup_sig_examples) < max_examples:
                    dup_sig_examples.append((sig, seen_sigs[sig], line_num))
            else:
                seen_sigs[sig] = line_num

            if out_handle and not (is_dup_id or is_dup_sig):
                out_handle.write(raw_line + "\n")
                written += 1
    finally:
        if out_handle:
            out_handle.close()

    result = {
        "path": path,
        "total": total,
        "parse_errors": parse_errors,
        "dup_id_total": dup_id_total,
        "dup_id_unique": len(dup_id_unique),
        "dup_sig_total": dup_sig_total,
        "dup_sig_unique": len(dup_sig_unique),
        "dup_id_examples": dup_id_examples,
        "dup_sig_examples": dup_sig_examples,
    }
    if output_path:
        result["written"] = written
        result["output_path"] = output_path
    return result


def find_files(root: Path, patterns: list[str]) -> list[Path]:
    files = set()
    for pattern in patterns:
        files.update(root.rglob(pattern))
    return sorted(files)


def default_output_path(path: Path) -> Path:
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-2:] == [".jsonl", ".gz"]:
        base = path.name[: -len(".jsonl.gz")]
        return path.with_name(f"{base}.deduped.jsonl.gz")
    if path.suffix == ".jsonl":
        return path.with_name(f"{path.stem}.deduped.jsonl")
    return path.with_name(f"{path.name}.deduped")


def print_report(result: dict, max_examples: int) -> None:
    rel_path = result["path"]
    print(f"\nFile: {rel_path}")
    print(f"  Rows: {result['total']}")
    if result["parse_errors"]:
        print(f"  Parse errors: {result['parse_errors']}")
    if result["dup_id_total"]:
        print(
            f"  Duplicate deck_id rows: {result['dup_id_total']} "
            f"(unique IDs: {result['dup_id_unique']})"
        )
        if result["dup_id_examples"]:
            print("  Deck ID examples:")
            for deck_id, first, dup in result["dup_id_examples"]:
                print(f"    {deck_id} (lines {first} and {dup})")
    if result["dup_sig_total"]:
        print(
            f"  Duplicate content rows: {result['dup_sig_total']} "
            f"(unique signatures: {result['dup_sig_unique']})"
        )
        if result["dup_sig_examples"]:
            print("  Content examples (signature hash):")
            for sig, first, dup in result["dup_sig_examples"]:
                print(f"    {sig} (lines {first} and {dup})")
    if result.get("output_path"):
        print(f"  Deduped output: {result['output_path']}")
        print(f"  Rows written: {result['written']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify duplicate decks in JSONL datasets."
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--file",
        help="Check a specific JSONL file (supports .jsonl and .jsonl.gz).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Write a deduped file, keeping the first occurrence.",
    )
    parser.add_argument(
        "--output",
        help="Output path for the deduped file (only with --dedupe).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with the deduped output.",
    )
    parser.add_argument(
        "--root",
        default=str(repo_root / "data"),
        help="Root folder to scan for JSONL datasets (default: ../data)",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=DEFAULT_PATTERNS,
        help="Glob pattern to include (can be repeated).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max duplicate examples to print per file.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit non-zero when duplicates are found.",
    )
    args = parser.parse_args()

    if args.in_place and not args.dedupe:
        parser.error("--in-place requires --dedupe")
    if args.output and not args.dedupe:
        parser.error("--output requires --dedupe")

    if args.file:
        files = [Path(args.file)]
    else:
        root = Path(args.root)
        files = find_files(root, args.pattern)

    if args.output and len(files) != 1:
        print("--output can only be used when checking a single file with --file")
        return 1

    if not files:
        if args.file:
            print(f"File not found: {args.file}")
        else:
            print(f"No JSONL files found under {root}")
        return 1

    any_dups = False
    for path in files:
        output_path = None
        temp_path = None
        if args.dedupe:
            if args.in_place:
                temp_path = path.with_name(path.name + ".dedupe.tmp")
                output_path = temp_path
            elif args.output:
                output_path = Path(args.output)
            else:
                output_path = default_output_path(path)
            if output_path.resolve() == path.resolve():
                print("Refusing to overwrite input without --in-place")
                return 1

        result = check_file(path, args.max_examples, output_path=output_path)
        if args.dedupe and args.in_place:
            temp_path.replace(path)
            result["output_path"] = path

        print_report(result, args.max_examples)
        if result["dup_id_total"] or result["dup_sig_total"]:
            any_dups = True

    if any_dups:
        print("\nDuplicates found.")
        return 0 if args.no_fail else 1

    print("\nOK: no duplicates found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
