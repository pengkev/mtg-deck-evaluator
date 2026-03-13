from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (BASE_DIR / "../data/general-decks.jsonl").resolve()
DEFAULT_OUTPUT = (BASE_DIR / "../data/general-vocabulary.txt").resolve()


def extract_names(board: object) -> Iterable[str]:
	if board is None:
		return []
	if isinstance(board, dict):
		return list(board.keys())
	if isinstance(board, list):
		names: list[str] = []
		for item in board:
			if isinstance(item, dict):
				if "name" in item:
					names.append(str(item["name"]))
					continue
				if "n" in item:
					names.append(str(item["n"]))
					continue
				card = item.get("card")
				if isinstance(card, dict) and "name" in card:
					names.append(str(card["name"]))
					continue
			elif isinstance(item, str):
				names.append(item)
		return names
	return []


def build_vocabulary(input_path: Path) -> tuple[set[str], int, int]:
	vocab: set[str] = set()
	total_lines = 0
	bad_lines = 0
	board_keys = ("mainboard", "sideboard", "commanders", "cmds", "main")

	with input_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			total_lines += 1
			line = line.strip()
			if not line:
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError:
				bad_lines += 1
				continue

			if not isinstance(record, dict):
				continue

			for key in board_keys:
				if key not in record:
					continue
				for name in extract_names(record.get(key)):
					if name:
						vocab.add(name)

	return vocab, total_lines, bad_lines


def main() -> int:
	parser = argparse.ArgumentParser(description="Build a card name vocabulary from general-decks.jsonl")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--no-sort", action="store_true")
	args = parser.parse_args()

	input_path = args.input.expanduser().resolve()
	output_path = args.output.expanduser().resolve()

	if not input_path.exists():
		raise SystemExit(f"Input not found: {input_path}")

	vocab, total_lines, bad_lines = build_vocabulary(input_path)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	names = list(vocab)
	if not args.no_sort:
		names.sort(key=str.casefold)

	output_path.write_text("\n".join(names) + "\n", encoding="utf-8")

	print(f"Read {total_lines} lines, skipped {bad_lines} bad lines")
	print(f"Wrote {len(names)} unique card names to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())