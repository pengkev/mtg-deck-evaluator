# MTG Deck Evaluator

> APS360 Final Project — Applied Fundamentals of Deep Learning, University of Toronto

Given a Magic: The Gathering decklist, predict its **power bracket** (1 = casual → 5 = fully competitive). Built for the **EDH** format: a 100-card singleton variant where decks are community-rated but no objective power metric exists.

Deck valuation is notoriously hard — even the game's designers struggle to maintain balance. Power emerges from subtle card synergies buried inside 100-card decks, not from any single card's strength. Static heuristics like deck price fail on Brackets 1–4 whose distributions nearly overlap. This project treats it as a **set classification problem**: a deck is a permutation-invariant bag of cards whose power is a latent function of co-occurrence patterns.

---

## Item2Vec Deepset Model Architecture

```
Decklist (up to 115 card IDs)
        ↓
Item2Vec Embedding Layer       ← 512-dim per card
        ↓
 Shared Card Encoder φ         ← Linear(512→128)
        ↓              ↓
  Card Features    Attention Net a(·)   ← scores from 128-dim features
    (128-dim)         (128→1 score)
        ↓              ↓
    Weighted Sum Pooling  Σ    ← permutation-invariant aggregation
        ↓
  Deck Classifier ρ (MLP)
        ↓
  Bracket (1–5)
```

Cards are embedded via **Item2Vec** (Word2Vec skip-gram), treating each deck as a "sentence" and each card as a "word." The classifier is an **attention-based Deep Sets** model whose weighted-sum pooling is permutation-invariant by design.

| Component      | Detail                                                        |
| -------------- | ------------------------------------------------------------- |
| Embedding      | 512-dim Item2Vec, vocab 10,037 cards                          |
| Card encoder φ | Linear(512→128), LayerNorm, ReLU, Linear(128→128), ReLU       |
| Attention net  | Linear(128→64), Tanh, Linear(64→1)                            |
| Deck MLP ρ     | Linear(128→128), BatchNorm, ReLU, Dropout(0.3), Linear(128→5) |
| Parameters     | ~5.25M trainable                                              |
| Optimizer      | AdamW; embeddings lr=1e-5, classifier lr=3e-4                 |
| Scheduler      | ReduceLROnPlateau (factor 0.5, patience 3)                    |

---

## Results

### Baseline: Price Threshold Classifier

Heuristic classifier using total deck price (USD) with four learned thresholds — no gradient descent. **36.5% accuracy** on 59,319 EDH decks. Correctly separates Bracket 5 (median $3,642) but fails on Brackets 1–4 (medians $476–$493, nearly identical distributions), confirming price alone is insufficient.

### Primary Model: Attention Deep Sets

Best peak validation accuracy: **23.98%** (LR 0.001, hidden 128, epoch 12) vs. 20% random chance.

Attention analysis on a Bracket 5 combo deck shows the model correctly upweights 0-mana cards (Gitaxian Probe, Memnite) as combo-density signals, but misses the actual win condition (Thassa's Oracle absent from top 10) — single-head pooling dilutes sparse, decisive synergies.

---

## Data

Decklists scraped from **Moxfield** (user self-tagged brackets 1–5) and **MTGTop8** (competitive tournament results). Raw data is not publicly shared.

**Embedding corpus:** 129,004 decks / 5.76M card tokens  
**Labelled set:** 44,496 EDH decks → 35,596 train / 4,449 val / 4,451 test

**Cleaning pipeline:**

1. Strip basic lands (universal filler, no strategic signal)
2. Merge sideboard into mainboard
3. Map card names → integer IDs via vocab lookup
4. Pad/truncate to fixed length 115

**Embedding quality** — nearest neighbors confirm semantic structure:

| Query         | Top neighbors                 |
| ------------- | ----------------------------- |
| Ponder        | Preordain, Brainstorm         |
| Demonic Tutor | Vampiric Tutor, Imperial Seal |
| Sol Ring      | Arcane Signet, Mana Vault     |

---

## Repository Layout

| Path                        | Purpose                                                  |
| --------------------------- | -------------------------------------------------------- |
| `data/`                     | Raw and processed datasets, model artifacts, checkpoints |
| `helpers/`                  | Data scraping and preprocessing scripts                  |
| `models/baseline/`          | Price-based baseline model and saved results             |
| `models/item2vec-deepsets/` | Primary deep learning model (notebook + weights)         |
| `models/set-transformer/`   | Placeholder for next architecture experiments            |

---

## Environment Setup

```bash
python -m venv .venv
```

**Windows:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

> **Note on working directories:** Several scripts use relative paths. Run utilities from `helpers/` and the baseline from `models/baseline/`.

---

## Common Workflows

### Build card → price mapping

Requires `data/AllIdentifiers.json` and `data/AllPrices.json`. These come from the MTGJSON project at https://mtgjson.com/

```bash
cd helpers && python cardtoprice.py
# Output: data/card_to_price.json
```

### Check data distribution

```bash
cd helpers && python distribution.py
# Outputs plots under data/
```

### Build card vocabulary

```bash
cd helpers && python vocabulary.py
# Output: data/general-vocabulary.txt
```

### Verify / deduplicate JSONL datasets

```bash
cd helpers
python verify_duplicates.py --root ../data
python verify_duplicates.py --file ../data/edh-decks.jsonl --dedupe  # write deduped file
```

### Run price baseline

```bash
cd models/baseline
python baseline.py \
  --prices ../../data/card_to_price.json \
  --jsonl  ../../data/edh-decks.jsonl \
  --output result.json

# Headless (no plots):
python baseline.py ... --no-viz
```

Outputs: console report, `result.json`, and distribution plots.

---

## Scraping Scripts

### Moxfield

> **API access:** Moxfield's API is undocumented and may break without notice. A valid user-agent must be obtained from Moxfield staff and set in a `.env` file at the repo root:
>
> ```env
> user-agent=YOUR_USER_AGENT_STRING
> ```

| Script                              | Purpose                                          |
| ----------------------------------- | ------------------------------------------------ |
| `helpers/moxfield-scraper-edh.py`   | Bracket-focused scraper, writes deck text files  |
| `helpers/large-moxfield-scraper.py` | Large-scale JSONL harvest with checkpoint resume |

### MTGTop8

| Script                                  | Purpose                                           |
| --------------------------------------- | ------------------------------------------------- |
| `helpers/mtgtop8-scraper-cEDH.py`       | Scrape competitive EDH results                    |
| `helpers/mtgtop8-scraper-general.py`    | Scrape general tournament results                 |
| `helpers/large-mtgtop8-scraper-cEDH.py` | Large-scale cEDH harvest                          |
| `helpers/enrich_decks_index.py`         | Add placement/player metadata to existing indices |

---

## Roadmap

- **Multi-head attention** (Transformer-style) to attend to distinct power axes simultaneously: mana efficiency, combo density, card advantage
- **Oracle text embeddings** — encode each card's rules text via a pretrained LM (e.g. BERT) to capture _what a card does_ rather than only co-play statistics; better generalises to underrepresented cards
- **Expanded dataset** — Moxfield community manager access granted; targeting low-to-mid six figures of labelled decklists
- **Community benchmark** — cross-validate predictions against CommanderSalt, EDHPowerLevel, DeckCheck CRISPI

---

## Acknowledgements

Special thanks to the team at [Moxfield](https://www.moxfield.com) for supporting this project by providing API access. Their support for academic data science work is greatly appreciated.

---

**Author:** Kevin Peng · `kev.peng@mail.utoronto.ca`  
**Course:** APS360, University of Toronto
