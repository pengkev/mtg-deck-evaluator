from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://archidekt.com"
OWNER_USERNAME = "GameKnights"
SEARCH_URL_CANDIDATES = (
    # Working server-side deck search endpoint used by Archidekt search page.
    f"{BASE_URL}/api/decks/v3/?ownerUsername={OWNER_USERNAME}&deckFormat=3&orderBy=-updatedAt&page={{page}}",
    # Fallback if deckFormat filtering changes server-side.
    f"{BASE_URL}/api/decks/v3/?ownerUsername={OWNER_USERNAME}&orderBy=-updatedAt&page={{page}}",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data").resolve()
OUTPUT_FILE = DATA_DIR / "gameknights_archidekt_decks.jsonl"
CHECKPOINT_FILE = DATA_DIR / "gameknights_archidekt_checkpoint.json"
WINNERS_FILE = DATA_DIR / "gameknights_winners.json"

RATE_LIMIT_DELAY_S = 0.25
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
MAX_SEARCH_PAGES = 200


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
    )

    # Optional: pass browser cookies from environment to reduce captcha friction.
    raw_cookie = os.getenv("ARCHIDEKT_COOKIE", "").strip()
    cf_clearance = os.getenv("ARCHIDEKT_CF_CLEARANCE", "").strip()
    if cf_clearance and "cf_clearance=" not in raw_cookie:
        raw_cookie = f"cf_clearance={cf_clearance}" + (f"; {raw_cookie}" if raw_cookie else "")

    if raw_cookie:
        session.headers["Cookie"] = raw_cookie

    return session


def request_headers(session: requests.Session) -> dict[str, str]:
    headers = {"User-Agent": session.headers.get("User-Agent", "Mozilla/5.0")}
    cookie = session.headers.get("Cookie")
    if isinstance(cookie, str) and cookie.strip():
        headers["Cookie"] = cookie.strip()
    return headers


def fetch_json(session: requests.Session, url: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT_S, headers=request_headers(session))
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if "json" not in content_type:
                preview = response.text[:120].replace("\n", " ")
                raise ValueError(f"Non-JSON response ({content_type}): {preview}")
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # Do not retry permanent client errors except throttling.
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
            last_error = exc
            backoff = attempt * 1.5
            logging.warning("Request failed (%s). Retrying in %.1fs: %s", attempt, backoff, url)
            time.sleep(backoff)
        except Exception as exc:
            last_error = exc
            backoff = attempt * 1.5
            logging.warning("Request failed (%s). Retrying in %.1fs: %s", attempt, backoff, url)
            time.sleep(backoff)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_iso_date(text: str | None) -> str | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return text


def discover_candidate_decks(
    session: requests.Session,
    max_pages: int = MAX_SEARCH_PAGES,
    delay_s: float = RATE_LIMIT_DELAY_S,
) -> list[dict]:
    discovered: dict[int, dict] = {}

    for template in SEARCH_URL_CANDIDATES:
        logging.info("Trying API search route: %s", template.format(page=1))
        route_had_results = False

        for page in range(1, max_pages + 1):
            url = template.format(page=page)
            try:
                data = fetch_json(session, url)
            except Exception as exc:
                logging.warning("Failed to fetch API search page %s: %s", page, exc)
                break

            results = data.get("results") or []
            if not results:
                if page == 1:
                    logging.info("No results from this API route on first page.")
                else:
                    logging.info("No decks found on search page %s; stopping.", page)
                break

            route_had_results = True
            new_on_page = 0
            for deck in results:
                deck_id = deck.get("id")
                if not isinstance(deck_id, int) or deck_id in discovered:
                    continue
                if bool(deck.get("private")):
                    continue

                discovered[deck_id] = {
                    "deck_id": deck_id,
                    "deck_url": f"{BASE_URL}/decks/{deck_id}",
                    "source_page": page,
                }
                new_on_page += 1

            logging.info(
                "Search page %s/%s: %s deck links (%s new)",
                page,
                max_pages,
                len(results),
                new_on_page,
            )

            if not data.get("next"):
                logging.info("Reached the last page of results.")
                break

            if delay_s:
                time.sleep(delay_s)

        if route_had_results and discovered:
            return list(discovered.values())

    logging.warning("API search routes returned no deck results; trying fallback user endpoint.")
    fallback = discover_recent_decks_fallback(session, OWNER_USERNAME)
    if fallback:
        logging.warning(
            "Fallback returned %s decks. This endpoint is often limited; search API is preferred.",
            len(fallback),
        )
    return fallback


def resolve_owner_id(session: requests.Session, username: str) -> int | None:
    try:
        response = session.get(
            f"{BASE_URL}/api/users/?username={username}",
            timeout=REQUEST_TIMEOUT_S,
            headers=request_headers(session),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.warning("Owner lookup failed for %s: %s", username, exc)
        return None
    results = payload.get("results") or []
    if not results:
        return None
    owner_id = results[0].get("id")
    return int(owner_id) if isinstance(owner_id, int) else None


def discover_recent_decks_fallback(session: requests.Session, username: str) -> list[dict]:
    owner_id = resolve_owner_id(session, username)
    if owner_id is None:
        return []

    try:
        response = session.get(
            f"{BASE_URL}/api/users/{owner_id}/decks/",
            timeout=REQUEST_TIMEOUT_S,
            headers=request_headers(session),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.warning("Fallback deck discovery failed for owner %s: %s", owner_id, exc)
        return []
    decks = payload.get("decks") or []
    out: list[dict] = []
    for deck in decks:
        deck_id = deck.get("id")
        is_private = bool(deck.get("private"))
        if not isinstance(deck_id, int):
            continue
        if is_private:
            continue
        out.append(
            {
                "deck_id": deck_id,
                "deck_url": f"{BASE_URL}/decks/{deck_id}",
                "name": deck.get("name"),
                "viewCount": deck.get("viewCount"),
                "private": is_private,
            }
        )
    return out


def parse_episode_key(deck_name: str) -> str | None:
    text = " ".join(deck_name.split())

    # Most GameKnights titles carry an internal episode code with a trailing player letter.
    code_match = re.search(r"\b([A-Z]{2,}\d{1,4})([A-Z])\b", text)
    if code_match:
        return code_match.group(1)

    explicit_patterns = (
        r"\bGame\s*Knights\s*#?\s*\d{1,4}\b",
        r"\bS\d{1,2}E\d{1,3}\b",
        r"\bEpisode\s*\d{1,3}\b",
        r"\bEp\.?\s*\d{1,3}\b",
        r"\bGK\s*#?\d{1,3}\b",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).upper().replace(" ", "")

    return None


def parse_archidekt_cards(deck_json: dict) -> tuple[list[dict], list[dict]]:
    main: list[dict] = []
    cmds: list[dict] = []

    for entry in deck_json.get("cards") or []:
        card_data = entry.get("card") or {}
        oracle = card_data.get("oracleCard") or {}
        name = oracle.get("name")
        qty = entry.get("quantity")
        if not isinstance(name, str) or not isinstance(qty, int):
            continue

        row = {"name": name, "qty": qty}
        categories = entry.get("categories") or []

        # Keep the same simplified schema as mtgdecks scraper:
        # commander(s) in cmds, the rest in main.
        if "Commander" in categories:
            cmds.append(row)
        elif "Sideboard" in categories or "Maybeboard" in categories:
            continue
        else:
            main.append(row)

    return main, cmds


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
    return {"processed_deck_ids": []}


def save_checkpoint(checkpoint_file: Path, processed_deck_ids: set[int]) -> None:
    payload = {
        "processed_deck_ids": sorted(processed_deck_ids),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_deck_id(deck_url: str) -> str:
    match = re.search(r"/decks/(\d+)", deck_url)
    if match:
        return match.group(1)
    return hashlib.sha1(deck_url.encode("utf-8")).hexdigest()[:12]


def load_winners(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logging.warning("Failed to parse winners file (%s): %s", path, exc)
    return {}


def is_winner_for_event(winner_rule: object, deck_id: int, deck_name: str) -> bool:
    if isinstance(winner_rule, int):
        return winner_rule == deck_id
    if isinstance(winner_rule, str):
        if winner_rule.isdigit():
            return int(winner_rule) == deck_id
        return winner_rule.lower() in deck_name.lower()
    if isinstance(winner_rule, list):
        return any(is_winner_for_event(item, deck_id, deck_name) for item in winner_rule)
    return False


def prompt_winner_for_event(event_key: str, event_decks: list[dict], winners_by_event: dict[str, object]) -> object | None:
    # Preserve already known winners unless user overrides interactively.
    existing = winners_by_event.get(event_key)

    print(f"\nEvent: Game Knights {event_key}")
    for idx, deck in enumerate(event_decks, start=1):
        print(f"  {idx}. [{deck['deck_id']}] {deck['deck_name']}")

    if existing is not None:
        print(f"Existing winner rule: {existing}")

    print("Pick winner by number, deck id, or text match.")
    print("Press Enter to keep existing winner; type 'skip' to mark no winner for this event.")

    while True:
        choice = input("Winner> ").strip()

        if not choice:
            return existing

        if choice.lower() == "skip":
            return None

        if choice.isdigit():
            value = int(choice)
            if 1 <= value <= len(event_decks):
                return event_decks[value - 1]["deck_id"]
            for deck in event_decks:
                if deck["deck_id"] == value:
                    return value

        lowered = choice.lower()
        for deck in event_decks:
            if lowered in deck["deck_name"].lower():
                return deck["deck_id"]

        print("Invalid winner selection. Try again.")


def scrape_gameknights_archidekt(
    delay_s: float = RATE_LIMIT_DELAY_S,
    max_pages: int = MAX_SEARCH_PAGES,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()

    seen_deck_urls = read_existing_deck_urls(OUTPUT_FILE)
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    processed_deck_ids = set(checkpoint.get("processed_deck_ids", []))
    winners_by_event = load_winners(WINNERS_FILE)

    candidate_decks = discover_candidate_decks(session, max_pages=max_pages, delay_s=delay_s)
    if not candidate_decks:
        logging.warning("No candidate decks discovered.")
        return 0

    deck_details: list[dict] = []
    for idx, deck_stub in enumerate(candidate_decks, start=1):
        deck_id = int(deck_stub["deck_id"])
        if deck_id in processed_deck_ids:
            continue

        deck_url = deck_stub["deck_url"]
        logging.info("[%s/%s] Deck: %s", idx, len(candidate_decks), deck_url)

        try:
            deck_json = fetch_json(session, f"{BASE_URL}/api/decks/{deck_id}/")
        except Exception as exc:
            logging.warning("Deck failed: %s (%s)", deck_url, exc)
            continue

        deck_name = deck_json.get("name")
        if not isinstance(deck_name, str):
            processed_deck_ids.add(deck_id)
            save_checkpoint(CHECKPOINT_FILE, processed_deck_ids)
            continue

        episode_key = parse_episode_key(deck_name)
        if not episode_key:
            processed_deck_ids.add(deck_id)
            save_checkpoint(CHECKPOINT_FILE, processed_deck_ids)
            continue

        main, cmds = parse_archidekt_cards(deck_json)
        if not main and not cmds:
            processed_deck_ids.add(deck_id)
            save_checkpoint(CHECKPOINT_FILE, processed_deck_ids)
            continue

        deck_details.append(
            {
                "deck_id": deck_id,
                "deck_url": deck_url,
                "deck_name": deck_name,
                "event_key": episode_key,
                "date": parse_iso_date(deck_json.get("updatedAt")),
                "main": main,
                "cmds": cmds,
            }
        )

        processed_deck_ids.add(deck_id)
        save_checkpoint(CHECKPOINT_FILE, processed_deck_ids)

        if delay_s:
            time.sleep(delay_s)

    if not deck_details:
        logging.info("No new GameKnights episode decks to write.")
        return 0

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in deck_details:
        grouped[row["event_key"]].append(row)

    new_records = 0
    with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
        for event_key, event_decks in grouped.items():
            players = len(event_decks)
            event_name = f"Game Knights {event_key}"
            winner_rule = prompt_winner_for_event(event_key, event_decks, winners_by_event)
            if winner_rule is None:
                winners_by_event.pop(event_key, None)
            else:
                winners_by_event[event_key] = winner_rule

            for deck in event_decks:
                if deck["deck_url"] in seen_deck_urls:
                    continue

                placement = 2
                if winner_rule is not None and is_winner_for_event(
                    winner_rule,
                    deck["deck_id"],
                    deck["deck_name"],
                ):
                    placement = 1

                record = {
                    "deck_id": parse_deck_id(deck["deck_url"]),
                    "deck_url": deck["deck_url"],
                    "event_url": SEARCH_URL_CANDIDATES[0].format(page=1),
                    "event_name": event_name,
                    "date": deck["date"],
                    "source": "archidekt-gameknights",
                    "placement": placement,
                    "players": players,
                    "placement_of": f"{placement}/{players}",
                    "main": deck["main"],
                    "cmds": deck["cmds"],
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_deck_urls.add(deck["deck_url"])
                new_records += 1

    WINNERS_FILE.write_text(json.dumps(winners_by_event, indent=2), encoding="utf-8")

    logging.info("Scrape complete. New decks written: %s", new_records)
    logging.info("Output file: %s", OUTPUT_FILE)
    return new_records


if __name__ == "__main__":
    scrape_gameknights_archidekt(
        delay_s=RATE_LIMIT_DELAY_S,
        max_pages=MAX_SEARCH_PAGES,
    )