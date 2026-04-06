from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urljoin
import re
import time
from datetime import datetime
import hashlib
from email.utils import parsedate_to_datetime
import logging
import xml.etree.ElementTree as ET

import cloudscraper
from bs4 import BeautifulSoup
import json

BASE_URL = "https://mtgdecks.net"
EVENTS_RSS_URL = f"{BASE_URL}/events/index.rss"
COMMANDER_TOURNAMENTS_URL = f"{BASE_URL}/Commander/tournaments"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data").resolve()
OUTPUT_FILE = DATA_DIR / "validation_decks.jsonl"
CHECKPOINT_FILE = DATA_DIR / "validation_checkpoint.json"

RATE_LIMIT_DELAY_S = 0.25
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
DEFAULT_ARCHIVE_PAGES = 150

PLAYERS_RE = re.compile(r"\[(\d[\d,]*)\s+Players?\]", re.IGNORECASE)
PLAYERS_FALLBACK_RE = re.compile(r"\b(\d[\d,]*)\s+Players?\b", re.IGNORECASE)
PLACEMENT_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
TOP_PLACEMENT_RE = re.compile(r"^Top\s*(\d+)\b", re.IGNORECASE)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def create_scraper() -> cloudscraper.CloudScraper:
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    scraper.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    return scraper


def fetch_text(scraper: cloudscraper.CloudScraper, url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = scraper.get(url, timeout=REQUEST_TIMEOUT_S)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            backoff = attempt * 1.5
            logging.warning("Request failed (%s). Retrying in %.1fs: %s", attempt, backoff, url)
            time.sleep(backoff)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_rss_commander_events(scraper: cloudscraper.CloudScraper) -> list[dict]:
    xml_text = fetch_text(scraper, EVENTS_RSS_URL)
    root = ET.fromstring(xml_text)

    events: list[dict] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        if "/Commander/" not in link:
            continue
        if "-tournament-" not in link:
            continue

        events.append({"title": title, "event_url": link, "pub_date": pub_date})

    return events


def parse_commander_tournaments_page_events(page_html: str) -> list[dict]:
    soup = BeautifulSoup(page_html, "html.parser")
    events_by_url: dict[str, dict] = {}

    for link in soup.find_all("a", href=True):
        raw_href = link.get("href")
        href = raw_href.strip() if isinstance(raw_href, str) else ""
        if "/Commander/" not in href:
            continue
        if "-tournament-" not in href:
            continue

        event_url = urljoin(BASE_URL, href)
        title = link.get_text(" ", strip=True) or None

        if event_url not in events_by_url:
            events_by_url[event_url] = {
                "title": title,
                "event_url": event_url,
                "pub_date": None,
            }

    return list(events_by_url.values())


def parse_archive_commander_events(
    scraper: cloudscraper.CloudScraper,
    max_pages: int = DEFAULT_ARCHIVE_PAGES,
    delay_s: float = RATE_LIMIT_DELAY_S,
) -> list[dict]:
    all_events: list[dict] = []
    seen_urls: set[str] = set()
    last_page_signature: tuple[str, ...] | None = None
    repeated_page_streak = 0

    for page in range(1, max_pages + 1):
        page_url = COMMANDER_TOURNAMENTS_URL if page == 1 else f"{COMMANDER_TOURNAMENTS_URL}/page:{page}"
        html = fetch_text(scraper, page_url)
        page_events = parse_commander_tournaments_page_events(html)

        if not page_events:
            logging.info("No tournament links found on archive page %s; stopping.", page)
            break

        page_urls = tuple(sorted(event["event_url"] for event in page_events))
        if page_urls == last_page_signature:
            repeated_page_streak += 1
        else:
            repeated_page_streak = 0
        last_page_signature = page_urls

        if repeated_page_streak >= 2:
            logging.info("Archive page content repeated at page %s; stopping.", page)
            break

        new_on_page = 0
        for event in page_events:
            event_url = event["event_url"]
            if event_url in seen_urls:
                continue
            seen_urls.add(event_url)
            all_events.append(event)
            new_on_page += 1

        logging.info(
            "Archive page %s/%s: %s events (%s new)",
            page,
            max_pages,
            len(page_events),
            new_on_page,
        )

        if delay_s:
            time.sleep(delay_s)

    return all_events


def discover_commander_events(
    scraper: cloudscraper.CloudScraper,
    max_archive_pages: int = DEFAULT_ARCHIVE_PAGES,
    delay_s: float = RATE_LIMIT_DELAY_S,
) -> list[dict]:
    events_by_url: dict[str, dict] = {}

    # RSS is useful for latest metadata (notably publication date).
    try:
        for event in parse_rss_commander_events(scraper):
            events_by_url[event["event_url"]] = event
    except Exception as exc:
        logging.warning("RSS discovery failed; continuing with archive pages (%s)", exc)

    for event in parse_archive_commander_events(scraper, max_pages=max_archive_pages, delay_s=delay_s):
        event_url = event["event_url"]
        if event_url in events_by_url:
            existing = events_by_url[event_url]
            if not existing.get("title") and event.get("title"):
                existing["title"] = event["title"]
            continue
        events_by_url[event_url] = event

    return list(events_by_url.values())


def decode_goto_path(goto_value: str | None) -> str | None:
    if not goto_value:
        return None
    try:
        decoded = base64.b64decode(goto_value + "===").decode("utf-8", errors="ignore")
    except Exception:
        return None
    return decoded if decoded.startswith("/") else None


def parse_players_from_event_page(event_soup: BeautifulSoup) -> int | None:
    text = " ".join(event_soup.stripped_strings)
    match = PLAYERS_RE.search(text)
    if match:
        return int(match.group(1).replace(",", ""))

    fallback = PLAYERS_FALLBACK_RE.search(text)
    if fallback:
        return int(fallback.group(1).replace(",", ""))

    return None


def placement_from_text(raw: str) -> int | None:
    cleaned = raw.replace("\xa0", " ").strip()
    first_token = cleaned.split("(", 1)[0].strip()

    match = PLACEMENT_RE.search(first_token)
    if match:
        return int(match.group(1))

    top_match = TOP_PLACEMENT_RE.search(first_token)
    if top_match:
        return int(top_match.group(1))

    return None


def parse_event_deck_entries(event_soup: BeautifulSoup) -> list[dict]:
    entries: list[dict] = []

    for row in event_soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        placement_text = cells[0].get_text(" ", strip=True)
        if not placement_text or placement_text == "Other":
            continue

        deck_path: str | None = None
        for node in row.find_all(attrs={"goto": True}):
            goto_value = node.get("goto")
            decoded = decode_goto_path(goto_value if isinstance(goto_value, str) else None)
            if not decoded:
                continue
            if not decoded.startswith("/Commander/"):
                continue
            if "decklist-by-" not in decoded:
                continue
            if decoded.endswith("/visual"):
                continue
            deck_path = decoded
            break

        if not deck_path:
            continue

        deck_url = urljoin(BASE_URL, deck_path)
        entries.append(
            {
                "deck_url": deck_url,
                "placement_text": placement_text,
                "placement": placement_from_text(placement_text),
            }
        )

    return entries


def parse_arena_deck_text(arena_text: str) -> tuple[list[dict], list[dict]]:
    main: list[dict] = []
    cmds: list[dict] = []

    section = "main"
    qty_line_re = re.compile(r"^(\d+)\s+(.+)$")

    for raw_line in arena_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        # MTGDecks arena export uses a Commander header before the main Deck header.
        if lower in {"commander", "commanders"}:
            section = "cmds"
            continue
        if lower in {"deck", "maindeck"}:
            section = "main"
            continue
        if lower in {"sideboard", "companion", "maybeboard"}:
            # Commander is stored in cmds; ignore other sections for this schema.
            section = "ignore"
            continue

        match = qty_line_re.match(line)
        if not match:
            continue

        qty = int(match.group(1))
        name = match.group(2).strip()
        if section == "ignore":
            continue
        target = main if section == "main" else cmds
        target.append({"name": name, "qty": qty})

    return main, cmds


def parse_deck_id(deck_url: str) -> str:
    # Most MTGDecks URLs end with a numeric deck id; fallback to URL hash if absent.
    match = re.search(r"-(\d+)(?:/)?$", deck_url)
    if match:
        return match.group(1)
    return hashlib.sha1(deck_url.encode("utf-8")).hexdigest()[:12]


def read_existing_deck_urls(output_file: Path) -> set[str]:
    seen: set[str] = set()
    if not output_file.exists():
        return seen

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                deck_url = obj.get("deck_url")
                if isinstance(deck_url, str):
                    seen.add(deck_url)
            except Exception:
                continue
    return seen


def load_checkpoint(checkpoint_file: Path) -> dict:
    if checkpoint_file.exists():
        try:
            return json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed_events": []}


def save_checkpoint(checkpoint_file: Path, processed_events: set[str]) -> None:
    payload = {
        "processed_events": sorted(processed_events),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_pub_date(pub_date: str | None) -> str | None:
    if not pub_date:
        return None
    try:
        dt = parsedate_to_datetime(pub_date)
        return dt.date().isoformat()
    except Exception:
        return pub_date


def scrape_validation_decks(
    max_events: int | None = None,
    delay_s: float = RATE_LIMIT_DELAY_S,
    max_archive_pages: int = DEFAULT_ARCHIVE_PAGES,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scraper = create_scraper()

    seen_deck_urls = read_existing_deck_urls(OUTPUT_FILE)
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    processed_events = set(checkpoint.get("processed_events", []))

    events = discover_commander_events(scraper, max_archive_pages=max_archive_pages, delay_s=delay_s)
    if max_events is not None:
        events = events[:max_events]

    logging.info("Commander events discovered (RSS + archive): %s", len(events))
    logging.info("Existing unique decks in output: %s", len(seen_deck_urls))

    new_records = 0

    for idx, event in enumerate(events, start=1):
        event_url = event["event_url"]
        event_title = event["title"]

        if event_url in processed_events:
            continue

        logging.info("[%s/%s] Event: %s", idx, len(events), event_title)

        try:
            event_html = fetch_text(scraper, event_url)
            event_soup = BeautifulSoup(event_html, "html.parser")
            players = parse_players_from_event_page(event_soup)
            event_entries = parse_event_deck_entries(event_soup)

            if not event_entries:
                logging.info("No deck entries found for event: %s", event_url)
                processed_events.add(event_url)
                save_checkpoint(CHECKPOINT_FILE, processed_events)
                continue

            with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
                for entry in event_entries:
                    deck_url = entry["deck_url"]
                    if deck_url in seen_deck_urls:
                        continue

                    try:
                        deck_html = fetch_text(scraper, deck_url)
                        deck_soup = BeautifulSoup(deck_html, "html.parser")

                        arena_textarea = deck_soup.find("textarea", id="arena_deck")
                        if not arena_textarea:
                            continue

                        arena_text = arena_textarea.get_text("\n", strip=True)
                        main, cmds = parse_arena_deck_text(arena_text)
                        if not main and not cmds:
                            continue

                        placement = entry["placement"]
                        record = {
                            "deck_id": parse_deck_id(deck_url),
                            "deck_url": deck_url,
                            "event_url": event_url,
                            "event_name": event_title,
                            "date": parse_pub_date(event.get("pub_date")),
                            "source": "mtgdecks",
                            "placement": placement,
                            "players": players,
                            "placement_of": (
                                f"{placement}/{players}"
                                if placement is not None and players is not None
                                else None
                            ),
                            "main": main,
                            "cmds": cmds,
                        }

                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        seen_deck_urls.add(deck_url)
                        new_records += 1

                        if delay_s:
                            time.sleep(delay_s)

                    except Exception as deck_exc:
                        logging.warning("Deck failed: %s (%s)", deck_url, deck_exc)
                        continue

            processed_events.add(event_url)
            save_checkpoint(CHECKPOINT_FILE, processed_events)

        except Exception as event_exc:
            logging.warning("Event failed: %s (%s)", event_url, event_exc)
            continue

    logging.info("Scrape complete. New decks written: %s", new_records)
    logging.info("Output file: %s", OUTPUT_FILE)
    return new_records


if __name__ == "__main__":
    # This validation scrape reads Commander events from RSS and paginated archives.
    scrape_validation_decks(
        max_events=None,
        delay_s=RATE_LIMIT_DELAY_S,
        max_archive_pages=DEFAULT_ARCHIVE_PAGES,
    )
