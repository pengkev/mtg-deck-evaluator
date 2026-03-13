import json
import matplotlib.pyplot as plt
from collections import Counter


def analyze_jsonl(file_path):
    """Analyze a JSONL file and return distribution statistics."""
    source_counts = Counter()
    deck_count = 0
    card_counts = []
    
    def get_source_bucket(source):
        s = str(source).lower()
        if 'tappedout' in s:
            return 'tappedout'
        if 'cedh' in s:
            return 'cedh'
        for bracket in ('1', '2', '3', '4', '5'):
            if f'bracket-{bracket}' in s or f'bracket_{bracket}' in s or f'bracket {bracket}' in s:
                return bracket
        return 'unknown'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                deck = json.loads(line)
                deck_count += 1
                source = get_source_bucket(deck.get('source', 'unknown'))
                source_counts[source] += 1
                
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
    title_fontsize = 18
    label_fontsize = 14
    tick_fontsize = 12
    value_fontsize = 11

    def format_source_label(source, max_len=16):
        label = str(source).lower().replace("_", " ").replace("-", " ")
        if label in {'cedh', 'unknown', 'tappedout'}:
            return label
        if label.isdigit():
            return label
        if len(label) > max_len:
            return label[: max_len - 3] + "..."
        return label
    
    def source_sort_key(item):
        source = str(item[0]).lower()
        if source.isdigit():
            return (0, int(source))
        if source == 'cedh':
            return (1, 0)
        if source == 'unknown':
            return (2, 0)
        if source == 'tappedout':
            return (3, 0)
        return (4, source)
    
    # Plot 1: Decks by source/bracket
    sources = list(stats['source_distribution'].keys())
    counts = list(stats['source_distribution'].values())
    
    # Sort by bracket number if applicable
    sorted_data = sorted(zip(sources, counts), key=source_sort_key)
    sorted_data = [item for item in sorted_data if str(item[0]).lower() != 'tappedout']
    sources, counts = zip(*sorted_data) if sorted_data else ([], [])
    display_labels = [format_source_label(source) for source in sources]
    x_positions = list(range(len(display_labels)))
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis([i / len(sources) for i in range(len(sources))])
    bars = ax1.bar(x_positions, counts, color=colors)
    ax1.set_xlabel('Source/Bracket', fontsize=label_fontsize)
    ax1.set_ylabel('Number of Decks', fontsize=label_fontsize)
    ax1.set_title(f'{title} - Deck Distribution by Source', fontsize=title_fontsize)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(display_labels, rotation=25, ha='right', fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)
    
    # Add count labels on bars
    y_offset = max(counts) * 0.01 if counts else 1
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                 f'{count:,}', ha='center', va='bottom', fontsize=value_fontsize)
    
    plt.tight_layout()
    plt.savefig(f'../data/{base_name}_by_source.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ../data/{base_name}_by_source.png")
    
    # Plot 2: Card count distribution histogram
    if stats['card_counts']:
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.hist(stats['card_counts'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Cards in Mainboard', fontsize=label_fontsize)
        ax2.set_ylabel('Number of Decks', fontsize=label_fontsize)
        ax2.set_title(f'{title} - Mainboard Size Distribution', fontsize=title_fontsize)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2.axvline(x=100, color='red', linestyle='--', label='Standard EDH (100)')
        ax2.legend(fontsize=tick_fontsize)
        
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
    
    def source_sort_key(item):
        source = str(item[0]).lower()
        if source.isdigit():
            return (0, int(source))
        if source == 'cedh':
            return (1, 0)
        if source == 'unknown':
            return (2, 0)
        if source == 'tappedout':
            return (3, 0)
        return (4, source)
    
    for source, count in sorted(stats['source_distribution'].items(), key=source_sort_key):
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
        ("../data/edh-decks.jsonl", "Decks Distribution"),
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