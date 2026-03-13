import logging, time, json, cloudscraper, os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# CONFIG
# ==========================================
load_dotenv()

SECRET_USER_AGENT = os.getenv('user-agent')
if not SECRET_USER_AGENT:
    print("❌ ERROR: 'user-agent' not found in .env file!")
else:
    masked = f"{SECRET_USER_AGENT[:8]}...{SECRET_USER_AGENT[-8:]}"
    print(f"✅ CREDENTIALS LOADED: [{masked}]")

BASE_URL = "https://api2.moxfield.com"
RATE_LIMIT_DELAY = 1.15
BRACKETS = [1, 2, 3, 4, 5]
MAX_PAGES_PER_BUCKET = 100

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data").resolve()
VOCAB_FILE = DATA_DIR / "general-vocabulary.txt"

OUTPUT_FILE = DATA_DIR / "large-moxfield/official_harvest.jsonl"
CHECKPOINT_FILE = DATA_DIR / "large-moxfield/checkpoint.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
scraper.headers.update({
    "User-Agent": SECRET_USER_AGENT,
    "Referer": "https://www.moxfield.com/",
})

# ==========================================
# JSONL-BACKED DEDUPLICATION
# ==========================================
SEEN_IDS = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                SEEN_IDS.add(json.loads(line)["id"])
            except Exception:
                pass
logging.info(f"Loaded {len(SEEN_IDS)} unique deck IDs from JSONL.")

# ==========================================
# CHECKPOINT HELPERS
# ==========================================
def load_checkpoint() -> tuple:
    """Returns (card_idx, bracket_idx, page) as a comparable tuple."""
    if CHECKPOINT_FILE.exists():
        try:
            raw = json.loads(CHECKPOINT_FILE.read_text())
            return (
                raw.get("card_idx", 0),
                raw.get("bracket_idx", 0),
                raw.get("page", 1),
            )
        except Exception:
            pass
    return (0, 0, 1)

def save_checkpoint(card_idx: int, b_idx: int, page: int, meta: str = ""):
    data = {
        "card_idx": card_idx,
        "bracket_idx": b_idx,
        "page": page,
        "meta": meta,
    }
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))

# ==========================================
# CARD EXTRACTION
# ==========================================
def extract_cards(data: dict) -> dict:
    boards = data.get("boards", {})
    def parse(name):
        record = boards.get(name, {}).get("cards", {})
        return [{"n": c.get("card", {}).get("name"), "q": c.get("quantity", 1)}
                for c in record.values() if c.get("card", {}).get("name")]
    return {"mainboard": parse("mainboard"), "commanders": parse("commanders")}

# ==========================================
# MAIN HARVEST
# ==========================================
def run_official_harvest():
    if not VOCAB_FILE.exists():
        logging.error(f"Vocabulary not found at {VOCAB_FILE}")
        return

    vocabulary = [line.strip() for line in VOCAB_FILE.read_text().splitlines() if line.strip()]
    checkpoint_tuple = load_checkpoint()
    logging.info(f"Resuming from checkpoint: card={checkpoint_tuple[0]}, bracket={checkpoint_tuple[1]}, page={checkpoint_tuple[2]}")

    for card_idx, card_name in enumerate(vocabulary):
        for b_idx, bracket in enumerate(BRACKETS):

            if (card_idx, b_idx, 1) < (checkpoint_tuple[0], checkpoint_tuple[1], 1):
                continue

            logging.info(f"📂 BUCKET: '{card_name}' | B{bracket}")

            for page in range(1, MAX_PAGES_PER_BUCKET + 1):

                if (card_idx, b_idx, page) < checkpoint_tuple:
                    continue

                time.sleep(RATE_LIMIT_DELAY)

                params = {
                    "pageNumber": page,
                    "pageSize": 100,
                    "sortType": "updated",
                    "sortDirection": "descending",
                    "fmt": "commander",
                    "minBracket": bracket,
                    "maxBracket": bracket,
                    "q": f"mainboard:{card_name}",
                }

                try:
                    resp = scraper.get(f"{BASE_URL}/v2/decks/search", params=params, timeout=20)

                    if resp.status_code != 200:
                        logging.warning(f"⚠️ HTTP {resp.status_code} — skipping bucket.")
                        break

                    decks = resp.json().get("data", [])
                    if not decks:
                        logging.info("📭 No decks returned — end of bucket.")
                        break

                    new_decks = [d for d in decks if d.get("publicId") not in SEEN_IDS]
                    yield_pct = (len(new_decks) / len(decks)) * 100
                    logging.info(f"📄 P{page}: {len(decks)} total | {len(new_decks)} new ({yield_pct:.1f}% yield)")

                    if yield_pct < 1.0 and page <= 2:
                        logging.info("⏭️ Low yield on early page — skipping bucket.")
                        break

                    for d in new_decks:
                        deck_id = d.get("publicId")
                        if not deck_id:
                            continue

                        time.sleep(RATE_LIMIT_DELAY)
                        det_resp = scraper.get(f"{BASE_URL}/v3/decks/all/{deck_id}", timeout=20)

                        if det_resp.status_code == 200:
                            raw = det_resp.json()
                            record = {
                                "id": deck_id,
                                "name": raw.get("name"),
                                "user_bracket": raw.get("userBracket"),
                                "auto_bracket": raw.get("autoBracket"),
                                "hubs": raw.get("hubNames", []),
                                **extract_cards(raw),
                            }
                            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record) + "\n")
                            SEEN_IDS.add(deck_id)

                    save_checkpoint(
                        card_idx, b_idx, page + 1,
                        meta=f"{card_name} | B{bracket}"
                    )

                except Exception as e:
                    logging.error(f"❌ Error on page {page}: {e}")
                    time.sleep(10)

if __name__ == "__main__":
    run_official_harvest()