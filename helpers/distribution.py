import json
import matplotlib.pyplot as plt
from collections import Counter


def analyze_jsonl(file_path):
    """Analyze a JSONL file and return distribution statistics."""
    source_counts = Counter()
    deck_count = 0
    card_counts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                deck = json.loads(line)
                deck_count += 1
                source_counts[deck.get('source', 'unknown')] += 1
                
                # Count cards in mainboard
                mainboard = deck.get('mainboard', {})
                total_cards = sum(mainboard.values())
                card_counts.append(total_cards)
    
    return {
        'total_decks': deck_count,
        'source_distribution': dict(source_counts),
        'card_counts': card_counts
    }


def visualize_distribution(stats, title="JSONL Distribution"):
    """Create visualizations for the distribution statistics."""
    base_name = title.replace(" ", "_").lower()
    
    # Plot 1: Decks by source/bracket
    sources = list(stats['source_distribution'].keys())
    counts = list(stats['source_distribution'].values())
    
    # Sort by bracket number if applicable
    sorted_data = sorted(zip(sources, counts), key=lambda x: x[0])
    sources, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis([i / len(sources) for i in range(len(sources))])
    bars = ax1.bar(sources, counts, color=colors)
    ax1.set_xlabel('Source/Bracket')
    ax1.set_ylabel('Number of Decks')
    ax1.set_title(f'{title} - Deck Distribution by Source')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'../data/{base_name}_by_source.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ../data/{base_name}_by_source.png")
    
    # Plot 2: Card count distribution histogram
    if stats['card_counts']:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(stats['card_counts'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Cards in Mainboard')
        ax2.set_ylabel('Number of Decks')
        ax2.set_title(f'{title} - Mainboard Size Distribution')
        ax2.axvline(x=100, color='red', linestyle='--', label='Standard EDH (100)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'../data/{base_name}_mainboard_size.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: ../data/{base_name}_mainboard_size.png")


def print_summary(stats, file_name):
    """Print a summary of the distribution statistics."""
    print(f"\n{'='*50}")
    print(f"File: {file_name}")
    print(f"{'='*50}")
    print(f"Total Decks: {stats['total_decks']:,}")
    print(f"\nDistribution by Source:")
    print("-" * 40)
    
    for source, count in sorted(stats['source_distribution'].items()):
        pct = (count / stats['total_decks']) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")
    
    if stats['card_counts']:
        avg_cards = sum(stats['card_counts']) / len(stats['card_counts'])
        min_cards = min(stats['card_counts'])
        max_cards = max(stats['card_counts'])
        print(f"\nMainboard Statistics:")
        print("-" * 40)
        print(f"  Average cards: {avg_cards:.1f}")
        print(f"  Min cards: {min_cards}")
        print(f"  Max cards: {max_cards}")


if __name__ == "__main__":
    # Analyze JSONL files
    jsonl_files = [
        ("../data/decks.jsonl", "Decks Distribution"),
        ("../data/general-decks.jsonl", "General Decks Distribution"),
    ]
    
    for file_path, title in jsonl_files:
        try:
            print(f"\nAnalyzing {file_path}...")
            stats = analyze_jsonl(file_path)
            print_summary(stats, file_path)
            visualize_distribution(stats, title)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")