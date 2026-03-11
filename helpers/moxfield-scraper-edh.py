from __future__ import annotations
import logging
import random
import time
import cloudscraper
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
BASE_URL = "https://www.moxfield.com"
API_BASE = "https://api.moxfield.com/v2"

BRACKET = 5

DATASET_DIR = Path(f"../data/moxfield-edh-bracket-{str(BRACKET)}")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Threads: 4-8 is usually the sweet spot for a single IP
MAX_WORKERS = 6 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

def get_scraper():
    """Returns a new scraper instance for each thread to avoid session collision."""
    return cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )

def format_deck_to_txt(data: dict) -> str:
    output = []
    mainboard = data.get('mainboard', {})
    for name, info in mainboard.items():
        output.append(f"{info.get('quantity', 1)} {name}")
    
    output.append("\nsideboard")
    
    commanders = data.get('commanders', {})
    for name, info in commanders.items():
        output.append(f"{info.get('quantity', 1)} {name}")
        
    sideboard = data.get('sideboard', {})
    for name, info in sideboard.items():
        if name not in commanders:
            output.append(f"{info.get('quantity', 1)} {name}")
            
    return "\n".join(output)

def process_deck(deck_meta: dict):
    """The function each worker thread will execute."""
    deck_id = deck_meta.get('publicId')
    if not deck_id:
        return None
    
    file_path = DATASET_DIR / f"{deck_id}.txt"
    if file_path.exists():
        return None

    # Individual scraper for the thread
    thread_scraper = get_scraper()
    url = f"{API_BASE}/decks/all/{deck_id}"
    
    try:
        # Mimic a human 'stumble' onto the page
        time.sleep(random.uniform(0.05, 0.2)) 
        
        headers = {"Referer": f"{BASE_URL}/decks/{deck_id}"}
        response = thread_scraper.get(url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            txt_content = format_deck_to_txt(data)
            file_path.write_text(txt_content, encoding="utf-8")
            
            logging.info(f"Saved: {deck_id}")
            return {"id": deck_id, "name": deck_meta.get('name', 'N/A'), "url": f"{BASE_URL}/decks/{deck_id}"}
        else:
            logging.warning(f"Failed {deck_id}: Status {response.status_code}")
    except Exception as e:
        logging.error(f"Error fetching {deck_id}: {e}")
    
    return None

def scrape_battlecruiser_parallel(max_pages=1000):
    csv_path = DATASET_DIR / "moxfield_log.csv"
    discovery_scraper = get_scraper()
    
    # We use a context manager for the thread pool to manage cleanup
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for page in range(1, max_pages + 1):
            logging.info(f"--- Discovery Page {page} ---")
            
            params = {
                "q": f'bracket:"{str(BRACKET)}"', 
                "fmt": "commander", 
                "sort": "views", 
                "pageSize": 50,
                "pageNumber": page
            }
            
            try:
                response = discovery_scraper.get(f"{API_BASE}/decks/search", params=params, headers={"Referer": BASE_URL})
                if response.status_code != 200:
                    logging.warning(f"Discovery Page {page} failed. Sleeping 10s...")
                    time.sleep(10)
                    continue
                
                decks = response.json().get('data', [])
                if not decks:
                    logging.info("End of results reached.")
                    break

                # Submit all 50 decks from this page to the executor
                futures = [executor.submit(process_deck, d) for d in decks]
                
                # Wait for this page's batch to finish and log results
                page_results = []
                for future in futures:
                    res = future.result()
                    if res:
                        page_results.append(res)
                
                if page_results:
                    df = pd.DataFrame(page_results)
                    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

                # Small breather between discovery pages
                time.sleep(random.uniform(0.05,0.1))

            except Exception as e:
                logging.error(f"Critical error on discovery page {page}: {e}")
                break

if __name__ == "__main__":
    scrape_battlecruiser_parallel(max_pages=1000)