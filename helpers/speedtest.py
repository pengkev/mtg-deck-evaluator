import time
import json
import cloudscraper
import statistics

# --- CONFIGURATION ---
# Replace with a valid public deck ID from your logs
SAMPLE_DECK_ID = "tZ0tQurf5kKQxTY63pPM3A" 
REQUESTS_PER_VERSION = 5

def get_session():
    session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    # Using the "Pro" headers found in the aleqsd repo for stability
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Referer": "https://www.moxfield.com/",
    })
    return session

def test_speed():
    scraper = get_session()
    
    # Endpoints to compare
    endpoints = [
        ("v2 (Standard)", f"https://api.moxfield.com/v2/decks/all/{SAMPLE_DECK_ID}"),
        ("v3 (Pro/api2)", f"https://api2.moxfield.com/v3/decks/all/{SAMPLE_DECK_ID}")
    ]

    results = {}

    for name, url in endpoints:
        print(f"Testing {name}...")
        times = []
        sizes = []
        
        for i in range(REQUESTS_PER_VERSION):
            start = time.perf_counter()
            resp = scraper.get(url, timeout=20)
            end = time.perf_counter()
            
            if resp.status_code == 200:
                times.append(end - start)
                sizes.append(len(resp.content))
            else:
                print(f"  Request {i+1} failed with status {resp.status_code}")
            
            # Small rest to avoid rate limiting during the test
            time.sleep(1)

        if times:
            results[name] = {
                "avg_time": statistics.mean(times),
                "avg_size_kb": statistics.mean(sizes) / 1024
            }

    print("\n" + "="*40)
    print(f"SPEED COMPARISON (ID: {SAMPLE_DECK_ID})")
    print("="*40)
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Avg Response Time: {data['avg_time']:.4f}s")
        print(f"  Avg Payload Size:  {data['avg_size_kb']:.2f} KB")
    print("="*40)

if __name__ == "__main__":
    test_speed()