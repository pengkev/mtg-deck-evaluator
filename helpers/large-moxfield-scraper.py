from __future__ import annotations
import logging, random, time, json, cloudscraper, os, threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# --- PRO CONFIGURATION ---
BASE_URL = "https://api2.moxfield.com" #
MAX_WORKERS = 6         # Slow and steady to avoid permanent bans
DAYS_TO_SCROLL = 1095   # 3 Years
BRACKETS = [1, 2, 3, 4, 5]
# Adding CI slices creates 16 search "buckets" per day/bracket
COLOR_SLICES = ["", "W", "U", "B", "R", "G", "WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]

OUTPUT_FILE = Path("../data/large-moxfield/master_harvest.jsonl")
CHECKPOINT_FILE = Path("../data/large-moxfield/checkpoint.txt")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

thread_local = threading.local()

def get_session():
    """Worker session with research-backed headers."""
    if not hasattr(thread_local, "session"):
        session = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Referer": "https://www.moxfield.com/",
        })
        thread_local.session = session
    return thread_local.session

# --- DUPLICATE PROTECTION ---
SEEN_IDS = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj: SEEN_IDS.add(obj["id"])
            except Exception: pass

def extract_deck_cards(data: dict) -> dict:
    """Navigates nested 'boards' and 'cards' mapping."""
    boards = data.get("boards", {})
    def parse(name: str):
        record = boards.get(name, {}).get("cards", {})
        return [{"name": c.get("card", {}).get("name"), "qty": c.get("quantity", 1)} 
                for c in record.values() if c.get("card", {}).get("name")]
    return {"main": parse("mainboard"), "cmds": parse("commanders")}

def process_deck(deck_meta: dict, bracket: int):
    deck_id = deck_meta.get('publicId')
    if not deck_id or deck_id in SEEN_IDS: return None
    scraper = get_session()
    try:
        # Use V3 for power-level metadata (autoBracket, userBracket)
        response = scraper.get(f"{BASE_URL}/v3/decks/all/{deck_id}", timeout=20)
        if response.status_code == 200:
            data = response.json()
            card_data = extract_deck_cards(data)
            if not card_data["main"] and not card_data["cmds"]: return None
            
            SEEN_IDS.add(deck_id)
            return {
                "id": deck_id, "name": data.get('name'), 
                "user_bracket": data.get('userBracket'), #
                "auto_bracket": data.get('autoBracket'), #
                **card_data
            }
    except Exception: pass
    return None

def start_harvest():
    discovery_scraper = get_session()
    start_offset = 1
    if CHECKPOINT_FILE.exists():
        start_offset = int(CHECKPOINT_FILE.read_text().strip())

    for day_offset in range(start_offset, DAYS_TO_SCROLL):
        target_date = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for bracket in BRACKETS:
            for color in COLOR_SLICES:
                color_q = f' identity:{color}' if color else ""
                logging.info(f"--- SWEEPING: {target_date} | B{bracket} | CI:{color or 'All'} ---")
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for page in range(1, 11): # 10 pages * 100 pageSize = 1k limit
                        params = {
                            "q": f'bracket:"{bracket}" updated:"{target_date}"{color_q}',
                            "fmt": "commander", "sort": "updated", "pageSize": 100, "pageNumber": page
                        }
                        try:
                            # Search-sfw provides the most stable index access
                            resp = discovery_scraper.get(f"{BASE_URL}/v2/decks/search-sfw", params=params, timeout=20)
                            decks = resp.json().get('data', []) if resp.status_code == 200 else []
                            if not decks: break 

                            futures = [executor.submit(process_deck, d, bracket) for d in decks]
                            batch = [f.result() for f in futures if f.result()]

                            if batch:
                                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                                    for r in batch: f.write(json.dumps(r) + "\n")
                                    f.flush(); os.fsync(f.fileno())
                                logging.info(f"Saved {len(batch)} new decks. (Vault: {len(SEEN_IDS)})")
                        except Exception: time.sleep(10)
        CHECKPOINT_FILE.write_text(str(day_offset))

if __name__ == "__main__":
    start_harvest()