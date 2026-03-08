"""
enrich_decks_index.py
---------------------
Reads an existing all_decks_index.csv (which lacks placement/players data)
and enriches it with placement, players, and placement_of by fetching each
deck's page on MTGTop8.

Usage:
    python enrich_decks_index.py

Output:
    Overwrites (or creates) all_decks_index_enriched.csv alongside the
    original file, with the three new columns added.
    When finished it replaces all_decks_index.csv with the enriched version.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://www.mtgtop8.com"
DATA_ROOT = Path("../data/mtgtop8-general")
INDEX_CSV = DATA_ROOT / "all_decks_index.csv"
ENRICHED_CSV = DATA_ROOT / "all_decks_index_enriched.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}

MAX_WORKERS = 4      # Parallel fetch threads
REQUEST_DELAY = 0.3  # Seconds between requests per thread (politeness)

# Regex patterns (same as the cEDH scraper)
PLACEMENT_RE = re.compile(
    r"(\d+)(?:st|nd|rd|th)\s+place\s*-\s*(\d+)\s+players?",
    re.IGNORECASE,
)
PLAYERS_RE = re.compile(r"(\d+)\s+players?\b", re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Thread-local session
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# Parsing helpers (ported from mtgtop8-scraper-cEDH.py)
# ---------------------------------------------------------------------------
def parse_deck_id(deck_url: str) -> str | None:
    query = urlparse(deck_url).query
    ids = parse_qs(query).get("d")
    return ids[0] if ids else None


def parse_players(soup: BeautifulSoup) -> int | None:
    text = " ".join(soup.stripped_strings)
    m = PLAYERS_RE.search(text)
    return int(m.group(1)) if m else None


def parse_placement_for_deck(soup: BeautifulSoup, deck_id: str | None) -> int | None:
    if not deck_id:
        return None
    for row in soup.select(".chosen_tr, .hover_tr"):
        link = row.find("a", href=True)
        if not link:
            continue
        row_deck_id = parse_deck_id(link["href"])
        if row_deck_id != deck_id:
            continue
        placement_text = row.find(class_="S14")
        if not placement_text:
            continue
        raw = placement_text.get_text(strip=True)
        if not raw:
            continue
        if "-" in raw:
            raw = raw.split("-", 1)[0].strip()
        try:
            return int(raw)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def enrich_row(row: pd.Series) -> dict:
    """Fetch the deck page and return a dict with placement/players fields."""
    deck_url: str = row["url"]
    deck_id = str(row["deck_id"])

    result = {
        "deck_id": row["deck_id"],
        "format": row["format"],
        "url": deck_url,
        "file": row["file"],
        "placement": None,
        "players": None,
        "placement_of": None,
    }

    session = get_session()

    for attempt in range(3):
        try:
            resp = session.get(deck_url, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logging.warning(f"Rate limited. Sleeping {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")

            placement = parse_placement_for_deck(soup, deck_id)
            players = parse_players(soup)

            result["placement"] = placement
            result["players"] = players
            result["placement_of"] = (
                f"{placement}/{players}"
                if placement is not None and players is not None
                else None
            )
            time.sleep(REQUEST_DELAY)
            break

        except Exception as exc:
            logging.warning(f"[{deck_id}] Attempt {attempt + 1} failed: {exc}")
            time.sleep(2 ** attempt)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not INDEX_CSV.exists():
        logging.error(f"Index file not found: {INDEX_CSV}")
        return

    df = pd.read_csv(INDEX_CSV, dtype=str)
    logging.info(f"Loaded {len(df)} rows from {INDEX_CSV}")

    # --- Resume support ---
    # If we already have a partial enriched file, skip rows already processed.
    already_done: set[str] = set()
    enriched_rows: list[dict] = []

    if ENRICHED_CSV.exists():
        done_df = pd.read_csv(ENRICHED_CSV, dtype=str)
        already_done = set(done_df["deck_id"].astype(str).tolist())
        logging.info(f"Resuming — {len(already_done)} rows already enriched.")
        # Keep already-done rows in memory so we can write a complete final file
        enriched_rows = done_df.to_dict("records")

    pending = df[~df["deck_id"].astype(str).isin(already_done)]
    logging.info(f"{len(pending)} rows still need enrichment.")

    if pending.empty:
        logging.info("Nothing to do — all rows are already enriched.")
    else:
        rows_list = [row for _, row in pending.iterrows()]
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(enrich_row, row): row for row in rows_list}

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                enriched_rows.append(result)
                completed += 1

                # Save checkpoint every 50 rows
                if completed % 50 == 0:
                    _write_checkpoint(enriched_rows)
                    logging.info(f"Checkpoint saved — {completed}/{len(rows_list)} done.")

        _write_checkpoint(enriched_rows)

    # --- Replace original index with enriched version ---
    final_df = pd.read_csv(ENRICHED_CSV)
    final_df.to_csv(INDEX_CSV, index=False)
    logging.info(f"all_decks_index.csv updated with placement/players columns.")
    logging.info(f"Enriched copy kept at: {ENRICHED_CSV}")


def _write_checkpoint(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(ENRICHED_CSV, index=False)


if __name__ == "__main__":
    main()
