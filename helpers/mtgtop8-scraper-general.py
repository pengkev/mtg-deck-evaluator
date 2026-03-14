from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
BASE_URL = "https://www.mtgtop8.com"
SEARCH_URL = "https://www.mtgtop8.com/search?current_page={page}"

# Target all major competitive formats
# ST=Standard, MO=Modern, LE=Legacy, VI=Vintage, cEDH=Commander, PI=Pioneer, PAU=Pauper
FORMATS = ["cEDH", "LE", "VI", "MO", "PI", "PAU", "ST"]

# Date range: ~10 Years (01/01/2016)
DATE_START = "01/01/2016"

# Root directory for data
DATA_ROOT = Path("../data/mtgtop8-general")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Thread-local storage for Sessions (Speed optimization)
thread_local = threading.local()

# Regex for parsing players/placement
PLACEMENT_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+place\s*-\s*(\d+)\s+players?", re.IGNORECASE)
PLAYERS_RE = re.compile(r"(\d+)\s+players?\b", re.IGNORECASE)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

def get_session():
    """Returns a thread-local session to reuse TCP connections."""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


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

def get_search_params(fmt: str, date_start: str) -> str:
    """Generates the specific query string for a format and date."""
    return (
        f"&event_titre=&deck_titre=&player=&format={fmt}"
        "&archetype_sel%5BVI%5D=&archetype_sel%5BLE%5D=&archetype_sel%5BMO%5D="
        "&archetype_sel%5BPI%5D=&archetype_sel%5BEX%5D=&archetype_sel%5BHI%5D="
        "&archetype_sel%5BST%5D=&archetype_sel%5BBL%5D=&archetype_sel%5BPAU%5D="
        "&archetype_sel%5BEDH%5D=&archetype_sel%5BHIGH%5D=&archetype_sel%5BEDHP%5D="
        "&archetype_sel%5BCHL%5D=&archetype_sel%5BPEA%5D=&archetype_sel%5BEDHM%5D="
        "&archetype_sel%5BALCH%5D=&archetype_sel%5BcEDH%5D=&archetype_sel%5BEXP%5D="
        "&archetype_sel%5BPREM%5D="
        "&compet_check%5BP%5D=1&compet_check%5BM%5D=1&compet_check%5BC%5D=1&compet_check%5BR%5D=1"
        f"&MD_check=1&cards=&date_start={date_start}&date_end="
    )

def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup | None:
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            logging.warning("Rate limit hit! Sleeping for 60s...")
            time.sleep(60)
            return fetch_soup(session, url)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def extract_deck_links(soup: BeautifulSoup) -> list[str]:
    links = []
    for row in soup.select(".hover_tr"):
        cells = row.find_all("td")
        if len(cells) < 2: continue
        link_tag = cells[1].find("a")
        if link_tag and link_tag.get("href"):
            links.append(link_tag["href"])
    return links

def parse_deck_id(deck_url: str) -> str | None:
    query = urlparse(deck_url).query
    ids = parse_qs(query).get("d")
    return ids[0] if ids else None

def process_deck(link: str, fmt: str, save_dir: Path) -> dict | None:
    """
    Worker function to process a single deck.
    Fetches the deck HTML page (for placement/players) and the MTGO text file.
    """
    session = get_session()

    try:
        safe_link = link if link.startswith("/") else f"/{link}"

        deck_url = urljoin(BASE_URL, link)
        deck_id = parse_deck_id(deck_url)

        if not deck_id:
            return None

        filename = f"{fmt}_{deck_id}.txt"
        file_path = save_dir / filename

        # --- RESUME CHECK ---
        if file_path.exists():
            return None

        # 1. Fetch deck HTML page to extract placement and player count
        placement: int | None = None
        players: int | None = None
        html_soup = fetch_soup(session, deck_url)
        if html_soup:
            placement = parse_placement_for_deck(html_soup, deck_id)
            players = parse_players(html_soup)

        # 2. Download MTGO text file
        mtgo_url = urljoin(BASE_URL, f"/mtgo{safe_link}")
        resp = session.get(mtgo_url, headers=HEADERS, timeout=20)

        if resp.status_code == 200:
            content = resp.text.replace("\r\n", "\n")
            # Sanity check: decks usually start with a digit
            if len(content) > 10 and any(c.isdigit() for c in content[:50]):
                file_path.write_text(content, encoding="utf-8")
                return {
                    "deck_id": deck_id,
                    "format": fmt,
                    "url": deck_url,
                    "file": str(file_path),
                    "placement": placement,
                    "players": players,
                    "placement_of": (
                        f"{placement}/{players}"
                        if placement is not None and players is not None
                        else None
                    ),
                }

    except Exception:
        pass

    return None

def scrape_format(fmt: str, max_pages=None):
    logging.info(f"--- STARTING FORMAT: {fmt} ---")
    
    fmt_dir = DATA_ROOT / fmt
    fmt_dir.mkdir(parents=True, exist_ok=True)
    
    # Master session for the pagination loop
    session = requests.Session()
    page = 0
    total_downloaded = 0
    
    while True:
        if max_pages and page > max_pages: break
        
        params = get_search_params(fmt, DATE_START)
        full_url = f"{SEARCH_URL.format(page=page)}{params}"
        
        soup = fetch_soup(session, full_url)
        if not soup: break
        
        links = extract_deck_links(soup)
        
        # Pagination Logic: If no links, or very few, we likely hit the end
        if not links:
            logging.info(f"[{fmt}] No decks on page {page}. Stopping.")
            break
            
        # 2. Process Decks in Parallel
        new_records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_deck, link, fmt, fmt_dir) for link in links]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    new_records.append(result)

        count = len(new_records)
        total_downloaded += count
        
        # Only log every 5 pages to reduce clutter, or if we found decks
        if count > 0 or page % 5 == 0:
            logging.info(f"[{fmt}] Page {page}: Saved {count} new decks. (Session Total: {total_downloaded})")
        
        # Update CSV incrementally
        if new_records:
            df = pd.DataFrame(new_records)
            csv_path = DATA_ROOT / "all_decks_index.csv"
            write_header = not csv_path.exists()
            df.to_csv(csv_path, mode='a', header=write_header, index=False)

        # Pagination Heuristic
        # If we found significantly fewer decks than the page size (usually 25 or 30), it's the last page.
        if len(links) < 5:
            logging.info(f"[{fmt}] Reached end of results at page {page}.")
            break

        page += 1
        time.sleep(1.0) # Politeness delay

if __name__ == "__main__":
    for fmt in FORMATS:
        if fmt == "ST": #standard rotation issues
            scrape_format(fmt, max_pages=50)                
        else:
            scrape_format(fmt, max_pages=None)