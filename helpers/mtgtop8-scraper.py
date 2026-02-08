from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.mtgtop8.com"
SEARCH_URL = "https://www.mtgtop8.com/search?current_page={page}"
SEARCH_PARAMS = (
    "&event_titre=&deck_titre=&player=&format=cEDH&archetype_sel%5BVI%5D="
    "&archetype_sel%5BLE%5D=&archetype_sel%5BMO%5D=&archetype_sel%5BPI%5D="
    "&archetype_sel%5BEX%5D=&archetype_sel%5BHI%5D=&archetype_sel%5BST%5D="
    "&archetype_sel%5BBL%5D=&archetype_sel%5BPAU%5D=&archetype_sel%5BEDH%5D="
    "&archetype_sel%5BHIGH%5D=&archetype_sel%5BEDHP%5D=&archetype_sel%5BCHL%5D="
    "&archetype_sel%5BPEA%5D=&archetype_sel%5BEDHM%5D=&archetype_sel%5BALCH%5D="
    "&archetype_sel%5BcEDH%5D=&archetype_sel%5BEXP%5D=&archetype_sel%5BPREM%5D="
    "&compet_check%5BP%5D=1&compet_check%5BM%5D=1&compet_check%5BC%5D=1"
    "&compet_check%5BR%5D=1&MD_check=1&cards=&date_start=&date_end="
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
    return BeautifulSoup(response.text, "html.parser")


def extract_deck_links(soup: BeautifulSoup) -> list[str]:
    links: list[str] = []
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


def scrape_decklists(max_pages: int | None = 5, delay_s: float = 0.0) -> pd.DataFrame:
    session = requests.Session()
    records: list[dict] = []
    seen_links: set[str] = set()

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break

        search_url = f"{SEARCH_URL.format(page=page)}{SEARCH_PARAMS}"
        soup = fetch_soup(session, search_url)
        page_links = extract_deck_links(soup)

        for link in page_links:
            if link in seen_links:
                continue
            seen_links.add(link)

            deck_url = urljoin(BASE_URL, link)
            deck_id = parse_deck_id(deck_url) or f"page{page}_idx{len(records)}"
            html_soup = fetch_soup(session, deck_url)
            players = parse_players(html_soup)
            placement = parse_placement_for_deck(html_soup, deck_id)

            mtgo_url = mtgo_export_url(link)
            mtgo_text = session.get(mtgo_url, headers=HEADERS, timeout=30).text

            deck_path = DATASET_DIR / f"deck_{deck_id}.txt"
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

        next_button = soup.select(".Nav_PN_no")
        has_next = bool(next_button and next_button[0].get_text(strip=True) == "Next")
        if not has_next:
            break

        page += 1

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    df = scrape_decklists(max_pages=5, delay_s=0.0)
    output_path = DATASET_DIR / "mtgtop8_decks.csv"
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} decks to {output_path}")
