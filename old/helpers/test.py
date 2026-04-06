import cloudscraper

def test_params():
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    # Using headers discovered in the repo
    scraper.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Referer": "https://www.moxfield.com/",
    })

    target_date = "2025-03-10"
    
    # Test 1: Your original syntax
    q1 = f'bracket:"1" updated:"{target_date}"'
    # Test 2: Potential shortened syntax
    q2 = f'bracket:"1" upd:"{target_date}"'

    for label, q in [("Original (updated)", q1), ("Shortened (upd)", q2)]:
        url = f"https://api2.moxfield.com/v2/decks/search-sfw?q={q}&pageSize=50"
        resp = scraper.get(url)
        count = len(resp.json().get('data', [])) if resp.status_code == 200 else "ERROR"
        print(f"{label} found: {count} decks")

if __name__ == "__main__":
    test_params()