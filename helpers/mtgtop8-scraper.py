from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
BASE_URL = "https://www.mtgtop8.com"
SEARCH_URL = "https://www.mtgtop8.com/search?current_page={page}"

# Calculate "Last 12 Months" date string (DD/MM/YYYY)
one_year_ago = datetime.now() - timedelta(days=365)
DATE_START = one_year_ago.strftime("%d/%m/%Y")

# We inject date_start={DATE_START} into the parameters
SEARCH_PARAMS = (
    "&event_titre=&deck_titre=&player=&format=cEDH&archetype_sel%5BVI%5D="
    "&archetype_sel%5BLE%5D=&archetype_sel%5BMO%5D=&archetype_sel%5BPI%5D="
    "&archetype_sel%5BEX%5D=&archetype_sel%5BHI%5D=&archetype_sel%5BST%5D="
    "&archetype_sel%5BBL%5D=&archetype_sel%5BPAU%5D=&archetype_sel%5BEDH%5D="
    "&archetype_sel%5BHIGH%5D=&archetype_sel%5BEDHP%5D=&archetype_sel%5BCHL%5D="
    "&archetype_sel%5BPEA%5D=&archetype_sel%5BEDHM%5D=&archetype_sel%5BALCH%5D="
    "&archetype_sel%5BcEDH%5D=&archetype_sel%5BEXP%5D=&archetype_sel%5BPREM%5D="
    "&compet_check%5BP%5D=1&compet_check%5BM%5D=1&compet_check%5BC%5D=1"
    f"&compet_check%5BR%5D=1&MD_check=1&cards=&date_start={DATE_START}&date_end="
)

DATASET_DIR = Path("../data/mtgtop8")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "mtg-deck-evaluator/1.0 (+https://www.mtgtop8.com)"}
PLACEMENT_RE = re.compile(
    r"(\d+)(?:st|nd|rd|th)\s+place\s*-\s*(\d+)\s+players?",
    re.IGNORECASE,
)
PLAYERS_RE = re.compile(r"(\d+)\s+players?\b", re.IGNORECASE)


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    # Fix encoding explicitly for MTGTop8
    response.encoding = response.apparent_encoding
    return BeautifulSoup(response.text, "html.parser")


def extract_deck_links(soup: BeautifulSoup) -> list[str]:
    links: list[str] = []
    # The search page uses 'hover_tr' for rows
    for row in soup.select(".hover_tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        link_tag = cells[1].find("a")
        if link_tag and link_tag.get("href"):
            links.append(link_tag["href"])
    return links


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


def scrape_decklists(max_pages: int | None = None, delay_s: float = 0.2) -> pd.DataFrame:
    session = requests.Session()
    records: list[dict] = []
    seen_links: set[str] = set()

    page = 1
    
    print(f"Starting Search for cEDH decks since {DATE_START}...")

    while True:
        if max_pages is not None and page > max_pages:
            break

        search_url = f"{SEARCH_URL.format(page=page)}{SEARCH_PARAMS}"
        print(f"Scraping Page {page}...")
        
        soup = fetch_soup(session, search_url)
        page_links = extract_deck_links(soup)

        if not page_links:
            print("No decks found on this page. Stopping.")
            break

        new_on_page = 0
        for link in page_links:
            if link in seen_links:
                continue
            seen_links.add(link)
            new_on_page += 1

            deck_url = urljoin(BASE_URL, link)
            deck_id = parse_deck_id(deck_url) or f"page{page}_idx{len(records)}"
            
            # FILE CHECK: Skip if we already have it
            deck_path = DATASET_DIR / f"deck_{deck_id}.txt"
            if deck_path.exists():
                # print(f"Skipping {deck_id} (already downloaded)")
                continue

            try:
                # 1. Fetch Deck HTML (for players/placement)
                html_soup = fetch_soup(session, deck_url)
                players = parse_players(html_soup)
                placement = parse_placement_for_deck(html_soup, deck_id)

                # 2. Fetch MTGO Text File
                mtgo_url = mtgo_export_url(link)
                mtgo_resp = session.get(mtgo_url, headers=HEADERS, timeout=30)
                mtgo_text = mtgo_resp.text.replace("\r\n", "\n") # Fix windows newlines

                deck_path.write_text(mtgo_text, encoding="utf-8")

                records.append(
                    {
                        "deck_id": deck_id,
                        "deck_url": deck_url,
                        "mtgo_url": mtgo_url,
                        "placement": placement,
                        "players": players,
                        "placement_of": f"{placement}/{players}"
                        if placement is not None and players is not None
                        else None,
                        "deck_file": str(deck_path),
                    }
                )

                if delay_s:
                    time.sleep(delay_s)
            except Exception as e:
                print(f"Error processing {deck_id}: {e}")
                continue

        # If we saw links but they were all duplicates, we might want to stop
        # But for safety in a search, we usually keep going unless the page was empty
        
        # Check for Next Button
        next_button = soup.select(".Nav_PN_no, .Nav_norm")
        has_next = False
        # The search page pagination is tricky, simpler to check if we got links
        if len(page_links) < 5: 
             # Heuristic: Search pages are usually 25 or 30 items. If we got very few, it's the end.
             has_next = False
        else:
             has_next = True

        # Save progress every page
        if records:
            partial_df = pd.DataFrame.from_records(records)
            output_path = DATASET_DIR / "mtgtop8_decks.csv"
            # Append if file exists, else write
            mode = 'a' if output_path.exists() and page > 1 else 'w'
            header = not output_path.exists() or page == 1
            partial_df.to_csv(output_path, mode=mode, header=header, index=False)
            records = [] # Clear memory after writing

        page += 1

    return pd.DataFrame() # Return empty since we wrote to CSV incrementally


if __name__ == "__main__":
    # max_pages=None to get everything
    scrape_decklists(max_pages=None, delay_s=0.2)
    print(f"Scraping complete. Check {DATASET_DIR}")