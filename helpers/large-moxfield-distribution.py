"""
Script to check the distribution of deck formats
"""

import json
from collections import Counter
from pathlib import Path


def check_format_distribution(filepath: str) -> None:
    """Count and display the distribution of deck formats."""
    format_counts = Counter()
    total_decks = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                deck = json.loads(line)
                if deck['user_bracket']:
                    format_name = deck.get('user_bracket')
                else:
                    format_name = deck['auto_bracket']
                format_counts[format_name] += 1
                total_decks += 1
    
    # Display results
    print(f"Total decks: {total_decks:,}")
    print(f"Number of brackets: {len(format_counts)}")
    print("\nFormat Distribution:")
    print("-" * 40)
    
    # Sort by count descending
    for format_name, count in format_counts.most_common():
        percentage = (count / total_decks) * 100
        print(f"{format_name} {count:>8,} ({percentage:5.2f}%)")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "large-moxfield" / "official_harvest.jsonl"
    check_format_distribution(str(data_path))
