from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os

# --- CONFIGURATION ---
BASE_URL = "https://www.mtgtop8.com"
SEARCH_URL = "https://www.mtgtop8.com/search"

# --- IMPROVED CONFIGURATION ---
today = datetime.now()
one_year_ago = today - timedelta(days=365)
two_years_ago = today - timedelta(days=730)
three_years_ago = today - timedelta(days=1095) # Target for pre-ban data

# primary window: 2 years ago -> 1 year ago
PRIMARY_START = two_years_ago.strftime("%d/%m/%Y")
PRIMARY_END = one_year_ago.strftime("%d/%m/%Y")

# legacy window (pre-ban): 3 years ago -> 2 years ago
LEGACY_START = three_years_ago.strftime("%d/%m/%Y")
LEGACY_END = two_years_ago.strftime("%d/%m/%Y")


DATASET_DIR = Path("../data/large-mtgtop8")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "mtg-deck-evaluator/1.0 (+https://www.mtgtop8.com)"}
PLACEMENT_RE = re.compile(
    r"(\d+)(?:st|nd|rd|th)\s+place\s*-\s*(\d+)\s+players?",
    re.IGNORECASE,
)
PLAYERS_RE = re.compile(r"(\d+)\s+players?\b", re.IGNORECASE)

MAX_WORKERS = 6
thread_local = threading.local()

def get_session():
    """Worker session to maintain connection pooling per thread."""
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        thread_local.session = session
    return thread_local.session


def fetch_soup(url: str, method: str = "get", data: dict | None = None) -> BeautifulSoup:
    session = get_session()
    if method.lower() == "post":
        response = session.post(url, data=data, timeout=30)
    else:
        response = session.get(url, timeout=30)
    response.raise_for_status()
    # Fix encoding explicitly for MTGTop8
    response.encoding = response.apparent_encoding
    return BeautifulSoup(response.text, "html.parser")


def extract_deck_info(soup: BeautifulSoup) -> list[dict]:
    results = []
    date_re = re.compile(r"\b\d{2}/\d{2}/\d{2}\b")
    # The search page uses 'hover_tr' for rows
    for row in soup.select(".hover_tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link_tag = cells[1].find("a")
        if link_tag and link_tag.get("href"):
            href = link_tag["href"]
            date_str = None
            for cell in cells:
                text = cell.get_text(strip=True)
                if date_re.search(text):
                    date_str = text
                    break
            results.append({"link": href, "date": date_str})
    return results


def parse_deck_id(deck_url: str) -> str | None:
    query = urlparse(deck_url).query
    deck_ids = parse_qs(query).get("d")
    return deck_ids[0] if deck_ids else None


def parse_players(soup: BeautifulSoup) -> int | None:
    text = " ".join(soup.stripped_strings)
    players_match = PLAYERS_RE.search(text)
    return int(players_match.group(1)) if players_match else None


def parse_placement_for_deck(soup: BeautifulSoup, deck_id: str | None) -> int | None:
    if not deck_id:
        return None
    for row in soup.select(".chosen_tr, .hover_tr"):
        link = row.find("a", href=True)
        if not link:
            continue
        link_deck_id = parse_deck_id(link["href"])
        if link_deck_id != deck_id:
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


def mtgo_export_url(deck_link: str) -> str:
    deck_path = deck_link if deck_link.startswith("/") else f"/{deck_link}"
    return urljoin(BASE_URL, f"/mtgo{deck_path}")

def parse_mtgo_text(mtgo_text: str) -> dict:
    main = []
    cmds = []
    current_board = main
    for line in mtgo_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() == "sideboard":
            current_board = cmds
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            current_board.append({"name": parts[1].strip(), "qty": int(parts[0])})
        else:
            # Fallback if no quantity prefix
            current_board.append({"name": line, "qty": 1})
    return {"main": main, "cmds": cmds}

OUTPUT_FILE = DATASET_DIR / "mtgtop8_decks.jsonl"
CHECKPOINT_FILE = DATASET_DIR / "checkpoint.txt"

SEEN_LINKS = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "deck_url" in obj: SEEN_LINKS.add(obj["deck_url"])
            except Exception: pass

def process_deck_link(link: str, deck_date: str | None, page: int, idx: int, delay_s: float) -> dict | None:
    deck_url = urljoin(BASE_URL, link)
    if deck_url in SEEN_LINKS:
        return None
        
    deck_id = parse_deck_id(deck_url) or f"page{page}_idx{idx}"

    try:
        # 1. Fetch Deck HTML (for players/placement)
        html_soup = fetch_soup(deck_url)
        players = parse_players(html_soup)
        placement = parse_placement_for_deck(html_soup, deck_id)

        # 2. Fetch MTGO Text File
        mtgo_url = mtgo_export_url(link)
        session = get_session()
        mtgo_resp = session.get(mtgo_url, timeout=30)
        mtgo_text = mtgo_resp.text.replace("\r\n", "\n") # Fix windows newlines
        
        parsed_cards = parse_mtgo_text(mtgo_text)

        if delay_s:
            time.sleep(delay_s)

        SEEN_LINKS.add(deck_url)
        return {
            "deck_id": deck_id,
            "deck_url": deck_url,
            "mtgo_url": mtgo_url,
            "date": deck_date,
            "placement": placement,
            "players": players,
            "placement_of": f"{placement}/{players}"
            if placement is not None and players is not None
            else None,
            "main": parsed_cards["main"],
            "cmds": parsed_cards["cmds"]
        }
    except Exception as e:
        print(f"Error processing {deck_id}: {e}")
        return None

def scrape_decklists(date_start: str, date_end: str, max_pages: int | None = None, delay_s: float = 0.2):
    # 2. Use a unique checkpoint for each date range
    checkpoint_path = DATASET_DIR / f"checkpoint_{date_start.replace('/', '-')}.txt"
    
    start_page = 1
    if checkpoint_path.exists():
        start_page = int(checkpoint_path.read_text().strip())
    
    page = start_page
    print(f"\n--- Initializing: {date_start} to {date_end} ---")

    while True:
        if max_pages is not None and page > max_pages:
            print(f"Reached page limit ({max_pages}). Moving to next task.")
            break

        print(f"Scraping Page {page}...")
        
        # Define local search parameters for THIS specific date range
        search_params = {
            "current_page": str(page),
            "event_titre": "",
            "deck_titre": "",
            "player": "",
            "format": "cEDH",
            "archetype_sel[VI]": "",
            "archetype_sel[LE]": "",
            "archetype_sel[MO]": "",
            "archetype_sel[PI]": "",
            "archetype_sel[EX]": "",
            "archetype_sel[HI]": "",
            "archetype_sel[ST]": "",
            "archetype_sel[BL]": "",
            "archetype_sel[PAU]": "",
            "archetype_sel[EDH]": "",
            "archetype_sel[HIGH]": "",
            "archetype_sel[EDHP]": "",
            "archetype_sel[CHL]": "",
            "archetype_sel[PEA]": "",
            "archetype_sel[EDHM]": "",
            "archetype_sel[ALCH]": "",
            "archetype_sel[cEDH]": "",
            "archetype_sel[EXP]": "",
            "archetype_sel[PREM]": "",
            "compet_check[P]": "1",
            "compet_check[M]": "1",
            "compet_check[C]": "1",
            "compet_check[R]": "1",
            "MD_check": "1",
            "cards": "",
            "date_start": date_start,
            "date_end": date_end
        }
        
        try:
            soup = fetch_soup(SEARCH_URL, method="post", data=search_params)
            page_items = extract_deck_info(soup)
        except Exception as e:
            print(f"Failed to fetch page {page}: {e}")
            break

        if not page_items:
            print("No more decks found for this date range.")
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_deck_link, item["link"], item["date"], page, idx, delay_s) 
                       for idx, item in enumerate(page_items)]
            batch = [f.result() for f in futures if f.result()]

        if batch:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for r in batch: f.write(json.dumps(r) + "\n")
            print(f"Saved {len(batch)} decks. Total unique decks: {len(SEEN_LINKS)}")
            
        # FIX: Save progress to the date-specific checkpoint
        checkpoint_path.write_text(str(page + 1))
        
        if len(page_items) < 5: # MTGTop8 usually has ~20-25 per page
             break
        page += 1

    return pd.DataFrame()

if __name__ == "__main__":
    # PASS 1: The Primary Data (Full Scrape)
    # Target: 2 years ago to 1 year ago
    print("--- PASS 1: PRIMARY DATA COLLECTION ---")
    scrape_decklists(PRIMARY_START, PRIMARY_END, max_pages=None, delay_s=0.2)

    # PASS 2: The Pre-Ban Data (Limited Scrape)
    # Target: Older data, but only the first 10 pages to prevent skewing the dataset
    print("\n--- PASS 2: LIMITED PRE-BAN DATA COLLECTION ---")
    scrape_decklists(LEGACY_START, LEGACY_END, max_pages=10, delay_s=0.3)
    
    print(f"All scraping tasks complete. Saved to {OUTPUT_FILE}")