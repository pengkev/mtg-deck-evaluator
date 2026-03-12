"""
Deck Pricing & Bracket Classification Baseline
Calculates deck prices using card_to_price.json (pre-computed 90-day averages)
and attempts to classify decks into brackets based solely on price.
"""

import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_card_prices(prices_path: str) -> dict:
    """Load card_to_price.json mapping."""
    print(f"Loading card prices from {prices_path}...")
    with open(prices_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def price_deck(deck: dict, card_prices: dict) -> dict:
    """
    Price a deck and return detailed breakdown.
    
    Args:
        deck: Dict with 'mainboard' and optionally 'sideboard' keys
        card_prices: Dict mapping card name -> price
    
    Returns:
        Dict with pricing breakdown
    """
    result = {
        'mainboard_total': 0.0,
        'sideboard_total': 0.0,
        'total': 0.0,
        'cards_priced': 0,
        'cards_missing': [],
        'breakdown': []
    }
    
    # Process mainboard
    mainboard = deck.get('mainboard', {})
    for card_name, quantity in mainboard.items():
        price = card_prices.get(card_name)
        
        if price is not None:
            card_total = price * quantity
            result['mainboard_total'] += card_total
            result['cards_priced'] += quantity
            result['breakdown'].append({
                'name': card_name,
                'quantity': quantity,
                'unit_price': round(price, 2),
                'total': round(card_total, 2),
                'location': 'mainboard'
            })
        else:
            result['cards_missing'].append(card_name)
    
    # Process sideboard
    sideboard = deck.get('sideboard', {})
    for card_name, quantity in sideboard.items():
        price = card_prices.get(card_name)
        
        if price is not None:
            card_total = price * quantity
            result['sideboard_total'] += card_total
            result['cards_priced'] += quantity
            result['breakdown'].append({
                'name': card_name,
                'quantity': quantity,
                'unit_price': round(price, 2),
                'total': round(card_total, 2),
                'location': 'sideboard'
            })
        else:
            if card_name not in result['cards_missing']:
                result['cards_missing'].append(card_name)
    
    result['total'] = result['mainboard_total'] + result['sideboard_total']
    result['mainboard_total'] = round(result['mainboard_total'], 2)
    result['sideboard_total'] = round(result['sideboard_total'], 2)
    result['total'] = round(result['total'], 2)
    
    # Sort breakdown by total price descending
    result['breakdown'].sort(key=lambda x: x['total'], reverse=True)
    
    return result


def print_price_report(result: dict):
    """Print a formatted price report."""
    print("\n" + "=" * 60)
    print("DECK PRICE REPORT")
    print("=" * 60)
    
    if 'deck_id' in result:
        print(f"Deck ID: {result['deck_id']}")
    if 'source' in result:
        print(f"Source: {result['source']}")
    
    print(f"\nTotal Deck Price: ${result['total']:,.2f}")
    print(f"  Mainboard: ${result['mainboard_total']:,.2f}")
    print(f"  Sideboard: ${result['sideboard_total']:,.2f}")
    print(f"\nCards Priced: {result['cards_priced']}")
    print(f"Cards Missing Price: {len(result['cards_missing'])}")
    
    if result['cards_missing']:
        print("\nMissing Prices:")
        for card in result['cards_missing'][:10]:
            print(f"  - {card}")
        if len(result['cards_missing']) > 10:
            print(f"  ... and {len(result['cards_missing']) - 10} more")
    
    print("\nTop 10 Most Expensive Cards:")
    print("-" * 50)
    for card in result['breakdown'][:10]:
        print(f"  {card['quantity']}x {card['name']}: ${card['total']:.2f} (${card['unit_price']:.2f} ea)")


def extract_bracket(source: str) -> int:
    """
    Extract bracket number from source string.
    - moxfield-edh-bracket-X -> bracket X
    - mtgtop8-cEDH -> bracket 5
    - tappedout-edh -> bracket 3 (assume mid-tier)
    """
    if 'bracket-' in source:
        try:
            return int(source.split('bracket-')[-1])
        except ValueError:
            return 3
    elif 'mtgtop8' in source.lower() or 'cedh' in source.lower():
        return 5
    elif 'tappedout' in source.lower():
        return 3  # Default assumption for unlabeled EDH
    return 3


def price_all_decks(jsonl_path: str, card_prices: dict, 
                    max_decks: int = None, verbose: bool = True) -> list[dict]:
    """
    Price all decks from a JSONL file.
    Returns list of dicts with price and bracket info.
    """
    results = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_decks and i >= max_decks:
                break
            
            if verbose and i % 1000 == 0:
                print(f"Processing deck {i}...")
            
            deck = json.loads(line)
            price_result = price_deck(deck, card_prices)
            
            results.append({
                'deck_id': deck.get('deck_id', f'deck_{i}'),
                'source': deck.get('source', 'unknown'),
                'bracket': extract_bracket(deck.get('source', '')),
                'price': price_result['total'],
                'cards_priced': price_result['cards_priced'],
                'cards_missing': len(price_result['cards_missing'])
            })
    
    return results


def compute_bracket_thresholds(results: list[dict]) -> dict:
    """
    Compute optimal price thresholds for bracket classification.
    Uses median prices per bracket to set boundaries.
    """
    # Group prices by bracket
    prices_by_bracket = defaultdict(list)
    for r in results:
        prices_by_bracket[r['bracket']].append(r['price'])
    
    # Calculate statistics per bracket
    stats = {}
    for bracket in sorted(prices_by_bracket.keys()):
        prices = prices_by_bracket[bracket]
        stats[bracket] = {
            'count': len(prices),
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices),
            'min': np.min(prices),
            'max': np.max(prices),
            'q25': np.percentile(prices, 25),
            'q75': np.percentile(prices, 75)
        }
    
    # Compute thresholds as midpoints between bracket medians
    brackets = sorted(stats.keys())
    thresholds = {}
    
    for i in range(len(brackets) - 1):
        b1, b2 = brackets[i], brackets[i + 1]
        # Use geometric mean of medians as threshold
        threshold = (stats[b1]['median'] + stats[b2]['median']) / 2
        thresholds[f'{b1}_to_{b2}'] = threshold
    
    return stats, thresholds


def predict_bracket_by_price(price: float, thresholds: dict) -> int:
    """
    Predict bracket based on price using learned thresholds.
    """
    # Extract threshold values in order
    t1 = thresholds.get('1_to_2', 200)
    t2 = thresholds.get('2_to_3', 400)
    t3 = thresholds.get('3_to_4', 700)
    t4 = thresholds.get('4_to_5', 1200)
    
    if price < t1:
        return 1
    elif price < t2:
        return 2
    elif price < t3:
        return 3
    elif price < t4:
        return 4
    else:
        return 5


def evaluate_price_baseline(results: list[dict], thresholds: dict) -> dict:
    """
    Evaluate how well price alone predicts bracket.
    """
    y_true = [r['bracket'] for r in results]
    y_pred = [predict_bracket_by_price(r['price'], thresholds) for r in results]
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    report = classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5], 
                                   target_names=['B1', 'B2', 'B3', 'B4', 'B5'],
                                   output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred
    }


def visualize_results(results: list[dict], stats: dict, thresholds: dict, 
                      eval_results: dict, output_dir: str = '../../data'):
    """Create visualizations for the price-based bracket analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Price distribution by bracket (box plot)
    prices_by_bracket = defaultdict(list)
    for r in results:
        prices_by_bracket[r['bracket']].append(r['price'])
    
    brackets = sorted(prices_by_bracket.keys())
    box_data = [prices_by_bracket[b] for b in brackets]
    
    bp = axes[0, 0].boxplot(box_data, labels=[f'B{b}' for b in brackets], patch_artist=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(brackets)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 0].set_xlabel('Bracket')
    axes[0, 0].set_ylabel('Deck Price ($)')
    axes[0, 0].set_title('Price Distribution by Bracket')
    axes[0, 0].set_yscale('log')
    
    # Add threshold lines
    threshold_values = sorted(thresholds.values())
    for t in threshold_values:
        axes[0, 0].axhline(y=t, color='blue', linestyle='--', alpha=0.5)
    
    # Plot 2: Price histogram by bracket
    for bracket in brackets:
        prices = prices_by_bracket[bracket]
        axes[0, 1].hist(prices, bins=50, alpha=0.5, label=f'Bracket {bracket}', 
                        density=True)
    axes[0, 1].set_xlabel('Deck Price ($)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Price Distribution Overlap')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 5000)
    
    # Plot 3: Confusion matrix
    cm = eval_results['confusion_matrix']
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 0].figure.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set(xticks=np.arange(5), yticks=np.arange(5),
                   xticklabels=['B1', 'B2', 'B3', 'B4', 'B5'],
                   yticklabels=['B1', 'B2', 'B3', 'B4', 'B5'],
                   xlabel='Predicted Bracket', ylabel='True Bracket',
                   title=f'Confusion Matrix (Acc: {eval_results["accuracy"]:.1%})')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(5):
        for j in range(5):
            axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    # Plot 4: Per-bracket metrics
    report = eval_results['classification_report']
    bracket_labels = ['B1', 'B2', 'B3', 'B4', 'B5']
    precisions = [report[b]['precision'] for b in bracket_labels]
    recalls = [report[b]['recall'] for b in bracket_labels]
    f1s = [report[b]['f1-score'] for b in bracket_labels]
    
    x = np.arange(len(bracket_labels))
    width = 0.25
    
    axes[1, 1].bar(x - width, precisions, width, label='Precision', color='steelblue')
    axes[1, 1].bar(x, recalls, width, label='Recall', color='coral')
    axes[1, 1].bar(x + width, f1s, width, label='F1-Score', color='seagreen')
    axes[1, 1].set_xlabel('Bracket')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Classification Metrics by Bracket')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(bracket_labels)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    plt.suptitle('Price-Based Bracket Classification Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_price_classification.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_baseline_report(stats: dict, thresholds: dict, eval_results: dict):
    """Print detailed baseline analysis report."""
    print("\n" + "=" * 70)
    print("PRICE-BASED BRACKET CLASSIFICATION BASELINE")
    print("=" * 70)
    
    print("\n📊 Price Statistics by Bracket:")
    print("-" * 70)
    print(f"{'Bracket':<10} {'Count':<10} {'Median':<12} {'Mean':<12} {'Std':<12} {'Range'}")
    print("-" * 70)
    for bracket in sorted(stats.keys()):
        s = stats[bracket]
        print(f"Bracket {bracket:<3} {s['count']:<10} ${s['median']:<11,.0f} ${s['mean']:<11,.0f} "
              f"${s['std']:<11,.0f} ${s['min']:.0f}-${s['max']:.0f}")
    
    print("\n🎯 Learned Price Thresholds:")
    print("-" * 50)
    for name, value in sorted(thresholds.items()):
        print(f"  {name}: ${value:,.2f}")
    
    print("\n📈 Classification Results:")
    print("-" * 50)
    print(f"  Overall Accuracy: {eval_results['accuracy']:.1%}")
    
    report = eval_results['classification_report']
    print(f"\n  Per-Bracket Performance:")
    print(f"  {'Bracket':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    for b in ['B1', 'B2', 'B3', 'B4', 'B5']:
        r = report[b]
        print(f"  {b:<10} {r['precision']:<12.2%} {r['recall']:<12.2%} "
              f"{r['f1-score']:<12.2%} {int(r['support'])}")
    
    print("\n💡 Key Insights:")
    print("-" * 50)
    
    # Find best and worst performing brackets
    f1_scores = {b: report[b]['f1-score'] for b in ['B1', 'B2', 'B3', 'B4', 'B5']}
    best = max(f1_scores, key=f1_scores.get)
    worst = min(f1_scores, key=f1_scores.get)
    
    print(f"  - Best predicted bracket: {best} (F1: {f1_scores[best]:.1%})")
    print(f"  - Worst predicted bracket: {worst} (F1: {f1_scores[worst]:.1%})")
    print(f"  - Price alone explains {eval_results['accuracy']:.1%} of bracket variance")
    
    if eval_results['accuracy'] < 0.4:
        print("  - Price is a WEAK predictor of bracket (significant overlap)")
    elif eval_results['accuracy'] < 0.6:
        print("  - Price is a MODERATE predictor of bracket")
    else:
        print("  - Price is a STRONG predictor of bracket")


def main():
    parser = argparse.ArgumentParser(description='Price-based bracket classification baseline')
    parser.add_argument('--prices', type=str, default='../../data/card_to_price.json',
                        help='Path to card_to_price.json')
    parser.add_argument('--jsonl', type=str, default='../../data/decks.jsonl',
                        help='Path to JSONL file containing decks')
    parser.add_argument('--max-decks', type=int, default=None,
                        help='Maximum number of decks to process (for testing)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Load card prices
    card_prices = load_card_prices(args.prices)
    print(f"Loaded prices for {len(card_prices)} cards")
    
    # Price all decks
    print(f"\nPricing decks from {args.jsonl}...")
    results = price_all_decks(args.jsonl, card_prices, max_decks=args.max_decks)
    
    print(f"\nProcessed {len(results)} decks")
    
    # Compute thresholds and evaluate
    stats, thresholds = compute_bracket_thresholds(results)
    eval_results = evaluate_price_baseline(results, thresholds)
    
    # Print report
    print_baseline_report(stats, thresholds, eval_results)
    
    # Visualize
    if not args.no_viz:
        visualize_results(results, stats, thresholds, eval_results)
    
    # Save results
    if args.output:
        output_data = {
            'stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                         for kk, vv in v.items()} for k, v in stats.items()},
            'thresholds': thresholds,
            'accuracy': eval_results['accuracy'],
            'classification_report': eval_results['classification_report'],
            'deck_results': results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
