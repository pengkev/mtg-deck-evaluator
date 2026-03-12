"""
Card Name to Price Mapping Generator
Creates a simple JSON mapping from card name -> average price (USD)
Uses 90-day average prices, selecting the cheapest printing for each card.
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict


def get_90_day_average(price_history: dict) -> float | None:
    """Calculate 90-day average price from price history."""
    if not price_history:
        return None
    
    reference_date = datetime.now()
    cutoff_date = reference_date - timedelta(days=90)
    
    prices = []
    for date_str, price in price_history.items():
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if cutoff_date <= date <= reference_date:
                prices.append(price)
        except (ValueError, TypeError):
            continue
    
    return sum(prices) / len(prices) if prices else None


def get_card_price(uuid: str, prices_data: dict) -> float | None:
    """Get the 90-day average price for a card UUID."""
    card_prices = prices_data.get(uuid, {})
    paper = card_prices.get('paper', {})
    
    # Try cardkingdom retail, then cardmarket retail
    for vendor in ['cardkingdom', 'cardmarket']:
        vendor_data = paper.get(vendor, {})
        retail = vendor_data.get('retail', {})
        
        # Try normal first, then foil
        for ptype in ['normal', 'foil']:
            history = retail.get(ptype, {})
            avg_price = get_90_day_average(history)
            if avg_price is not None:
                return avg_price
    
    return None


def main():
    # Load AllIdentifiers.json
    print("Loading AllIdentifiers.json...")
    with open('../data/AllIdentifiers.json', 'r', encoding='utf-8') as f:
        identifiers = json.load(f)
    
    # Load AllPrices.json
    print("Loading AllPrices.json...")
    with open('../data/AllPrices.json', 'r', encoding='utf-8') as f:
        prices = json.load(f)
    
    prices_data = prices['data']
    
    # Build name -> list of UUIDs mapping
    print("Building card name to UUID mapping...")
    name_to_uuids = defaultdict(list)
    for uuid, card_data in identifiers['data'].items():
        name = card_data.get('name', '')
        if name:
            name_to_uuids[name].append(uuid)
    
    # For each card name, find the cheapest printing price
    print(f"Processing {len(name_to_uuids)} unique card names...")
    card_to_price = {}
    missing_count = 0
    
    for i, (name, uuids) in enumerate(name_to_uuids.items()):
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(name_to_uuids)} cards...")
        
        best_price = None
        for uuid in uuids:
            price = get_card_price(uuid, prices_data)
            if price is not None:
                if best_price is None or price < best_price:
                    best_price = price
        
        if best_price is not None:
            card_to_price[name] = round(best_price, 2)
        else:
            missing_count += 1
    
    # Save result
    output_path = '../data/card_to_price.json'
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(card_to_price, f, indent=2)
    
    print(f"\nDone!")
    print(f"  Cards with prices: {len(card_to_price)}")
    print(f"  Cards missing prices: {missing_count}")
    
    # Show some examples
    print("\nSample prices:")
    samples = ['Sol Ring', 'Command Tower', 'Mana Crypt', 'Black Lotus', 'Lightning Bolt']
    for card in samples:
        if card in card_to_price:
            print(f"  {card}: ${card_to_price[card]:.2f}")


if __name__ == "__main__":
    main()