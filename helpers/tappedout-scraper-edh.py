from __future__ import annotations

import logging
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

import pandas as pd
import requests

# --- CONFIGURATION ---
BASE_URL = "https://tappedout.net"

DATASET_DIR = Path("../data/tappedout-edh")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "mtg-deck-evaluator/1.0 (+https://github.com/kevp9/mtg-deck-evaluator)"}

# Regex to extract the deck slug from a TappedOut URL
SLUG_RE = re.compile(r"tappedout\.net/mtg-decks/([^/?#]+)")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")


def fetch_txt(session: requests.Session, url: str) -> str | None:
    """Fetch a URL and return its text content, handling rate limits."""
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            logging.warning("Rate limit hit. Sleeping 60s...")
            time.sleep(60)
            return fetch_txt(session, url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None


def parse_slug(url: str) -> str | None:
    """Extract the deck slug from a TappedOut URL."""
    match = SLUG_RE.search(url)
    return match.group(1) if match else None


def parse_deck_txt(raw: str) -> tuple[str, str | None]:
    """
    Converts the raw TappedOut /?fmt=txt response into MTGO format:
      - Mainboard cards as-is
      - Commander (*CMDR*) moved to a Sideboard section

    Returns (deck_text, commander_name).
    """
    mainboard: list[str] = []
    sideboard: list[str] = []
    commander: str | None = None

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        if "*CMDR*" in line:
            clean = line.replace("*CMDR*", "").strip()
            sideboard.append(clean)
            # Strip the leading count to get just the card name
            parts = clean.split(None, 1)
            if len(parts) == 2:
                commander = parts[1]
        else:
            mainboard.append(line)

    sections = mainboard
    if sideboard:
        sections = mainboard + ["", "Sideboard"] + sideboard

    return "\n".join(sections), commander


def scrape_decklists(urls: list[str], delay_s: float = 1.5) -> pd.DataFrame:
    """
    Given a list of TappedOut deck URLs, download each as a .txt file
    (MTGO format) and write metadata to tappedout_decks.csv incrementally.

    Already-downloaded decks are skipped automatically (resume-safe).
    """
    session = requests.Session()

    logging.info(f"Starting TappedOut scrape: {len(urls)} deck(s) queued.")

    for i, url in enumerate(urls, 1):
        slug = parse_slug(url)
        if not slug:
            logging.warning(f"[{i}/{len(urls)}] Could not parse slug from: {url} — skipping.")
            continue

        deck_path = DATASET_DIR / f"deck_{slug}.txt"

        # --- RESUME CHECK ---
        if deck_path.exists():
            logging.info(f"[{i}/{len(urls)}] Skipping {slug} (already downloaded)")
            continue

        clean_url = url.split("?")[0].rstrip("/")
        txt_url = f"{clean_url}/?fmt=txt"

        logging.info(f"[{i}/{len(urls)}] Fetching {slug}...")

        raw = fetch_txt(session, txt_url)
        if not raw:
            logging.warning(f"No content returned for {slug}.")
            continue

        deck_text, commander = parse_deck_txt(raw)

        # Sanity check: deck content should open with a card count
        if not deck_text or not any(c.isdigit() for c in deck_text[:50]):
            logging.warning(f"Unexpected content for {slug}, skipping.")
            continue

        deck_path.write_text(deck_text, encoding="utf-8")

        record = {
            "deck_id": slug,
            "deck_url": clean_url,
            "txt_url": txt_url,
            "commander": commander,
            "deck_file": str(deck_path),
        }

        # Save CSV progress after every deck so nothing is lost on interruption
        csv_path = DATASET_DIR / "tappedout_decks.csv"
        write_header = not csv_path.exists()
        pd.DataFrame([record]).to_csv(csv_path, mode="a", header=write_header, index=False)

        if delay_s:
            time.sleep(delay_s)

    logging.info(f"Scrape complete. Check {DATASET_DIR}")
    return pd.DataFrame()  # Written to CSV incrementally


def discover_deck_urls(num_pages=3, delay_s=3.0):
    """
    Crawls TappedOut search results to find deck URLs.
    Updated for 2026 HTML structure and bot mitigation.
    """
    base_search_url = "https://tappedout.net/mtg-decks/search/"
    deck_urls = []
    
    # More realistic headers to pass Cloudflare/TappedOut checks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://tappedout.net/"
    }

    session = requests.Session()
    # Initial hit to set cookies
    session.get("https://tappedout.net/", headers=headers)

    for page in range(1, num_pages + 1):
        params = {
            "format": "edh",
            "o": "-date_updated",
            "page": page
        }
        
        logging.info(f"Searching page {page}...")
        
        try:
            response = session.get(base_search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # THE FIX: TappedOut now uses 'name' or 'deck-link' classes 
            # We look for any link containing '/mtg-decks/' that isn't a user profile
            links = soup.find_all('a', href=re.compile(r'^/mtg-decks/[^/]+/$'))
            
            page_urls = []
            for link in links:
                full_url = f"https://tappedout.net{link['href']}"
                # Filter out common false positives like 'latest' or 'search'
                if "/mtg-decks/search/" not in full_url:
                    page_urls.append(full_url)
            
            # Deduplicate page results
            page_urls = list(set(page_urls))
            deck_urls.extend(page_urls)
            
            if not page_urls:
                logging.warning(f"Page {page} returned 0 results. You might be soft-blocked.")
            else:
                logging.info(f"Found {len(page_urls)} decks on page {page}.")
            
            time.sleep(delay_s) # 3.0s is safer for TappedOut
            
        except Exception as e:
            logging.error(f"Error on page {page}: {e}")
            break
            
    return list(set(deck_urls))

def discover_massive_deck_list(total_pages_per_color=3):
    """
    Refined discovery: Uses specific identity flags and tuple-based params 
    to ensure multi-color searches actually return results.
    """
    # Note: For Colorless, we use an empty list; for others, the letter string.
    color_codes = [
        "", "W", "U", "B", "R", "G", 
        "WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG",
        "WUB", "WUR", "WUG", "WBR", "WBG", "WRG", "UBR", "UBG", "URG", "BRG",
        "WUBR", "WUBG", "WURG", "WBRG", "UBRG", "WUBRG"
    ]
    
    all_found_urls = set()
    session = requests.Session()
    session.get(BASE_URL, headers=HEADERS)

    for colors in color_codes:
        logging.info(f"--- Searching Color Identity: {colors if colors else 'Colorless'} ---")
        
        for page in range(1, total_pages_per_color + 1):
            # 1. Build params as a list of tuples to handle duplicate 'c' keys perfectly
            payload = [
                ("format", "edh"),
                ("o", "-date_updated"),
                ("page", page),
                ("identity_basis", "identity"), # Force Color Identity mode
            ]
            
            # Add each color as a separate 'c' parameter
            for char in colors:
                payload.append(("c", char))
            
            try:
                # Use 'params=payload' with the list of tuples
                response = session.get(f"{BASE_URL}/mtg-decks/search/", params=payload, headers=HEADERS)
                
                # DEBUG: If you get 0 results, uncomment the line below to check the URL in your browser
                # logging.info(f"Checking URL: {response.url}")

                if response.status_code != 200:
                    logging.warning(f"Error {response.status_code}. Sleeping...")
                    time.sleep(10)
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # THE FIX: More aggressive link finding. 
                # TappedOut results are usually in a <table> with <td> tags.
                links = soup.select('td a[href^="/mtg-decks/"]')
                
                if not links:
                    # Fallback to broad search if selector fails
                    links = soup.find_all('a', href=re.compile(r'^/mtg-decks/[^/]+/$'))

                new_on_page = 0
                for link in links:
                    url = f"{BASE_URL}{link['href']}"
                    # Ensure it's a deck list and not a search/pagination link
                    if "/search/" not in url and "/latest/" not in url:
                        if url not in all_found_urls:
                            all_found_urls.add(url)
                            new_on_page += 1
                
                logging.info(f"Page {page} ({colors}): Found {new_on_page} new decks.")
                
                if not links: 
                    break # End of results for this color

                time.sleep(2.5)
                
            except Exception as e:
                logging.error(f"Error on {colors}: {e}")
                break
                
    return list(all_found_urls)

if __name__ == "__main__":
    # 1. Load existing IDs so we don't even try to scrape them
    existing_ids = set()
    csv_path = DATASET_DIR / "tappedout_decks.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        existing_ids = set(df['deck_id'].tolist())

    # 2. Find a huge batch of URLs
    # This will take a while but will find hundreds/thousands of unique decks
    urls = discover_massive_deck_list(total_pages_per_color=3) 
    
    # 3. Filter out URLs we already have in our CSV/Folder
    new_urls = [u for u in urls if parse_slug(u) not in existing_ids]
    
    logging.info(f"Discovery found {len(urls)} decks. {len(new_urls)} are new. Starting scrape...")
    
    # 4. Scrape only the new ones
    if new_urls:
        scrape_decklists(new_urls, delay_s=2.0)