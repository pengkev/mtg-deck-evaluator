# Magic: The Gathering Deck Power Level Evaluator

## 1. Initial Architecture & Discoveries

The original pipeline mapped cards via Word2Vec, aggregated them using a Deep Sets architecture, and utilized standard classification to predict 5 distinct power Brackets. Performance flatlined at ~20% (random guessing) due to severe data and labeling flaws:

- **The `<UNK>` Avalanche:** Inconsistent naming (split cards, MDFCs) mapped up to 15% of a deck's cards to the `<UNK>` token.
- **The Moxfield Heuristic Trap:** Neural networks seek deep mathematical synergy, but the loss function punished the model for missing Moxfield's rigid, rules-based "Game Changer" checklists.
- **The Human Ego Problem:** Users frequently miscalibrate their own power levels, labeling mid-power decks as cEDH.
- **Classification Blur:** Deep Sets aggregate cards in a vacuum, causing the model to struggle to separate B1 from B2 (detecting a single generic staple) and B4 from B5 (failing to understand how fast mana interacts with the rest of the shell).

## 2. The Solution: Hybrid Embeddings

To fix the vocabulary and mechanical blind spots, the pipeline shifted to an **896-dimensional Hybrid Latent Space**, fusing statistical meta-usage with actual game rules.

- **Word2Vec (512-D):** Trained on a 300k mixed-format megacorpus to capture deck archetypes and competitive synergies.
- **Domain-Adapted NLP (384-D):** The `microsoft/MiniLM-L12-H384-uncased` model was fine-tuned via Masked Language Modeling (MLM) on Scryfall Oracle text. Perplexity dropped from ~50,000 to 5, proving the model natively learned MTG jargon.
- **Universal Metadata Headers:** Prepended physical stats (e.g., `Name: Lightning Bolt. Mana cost: {R}. Type: Instant.`) to rules text so the NLP model inherently grasps card "rate" and mana-efficiency.
- **Pipeline Upgrades:** Implemented diacritic stripping, split-card deduplication, and Vanilla creature support (using type-lines and stats as text) to ensure zero data loss.

## 3. Architecture & Objective Pivot

To solve the Deep Sets "vacuum" problem and protect against bad labels, the core engine was overhauled:

- **Transformer Encoder:** Cards now update their own mathematical weights via Self-Attention. A _Lion's Eye Diamond_ alongside _Brain Freeze_ is flagged as a lethal combo, while an isolated LED is muted.
- **Ordinal Regression:** Power is a continuous spectrum. Predicting a single float (e.g., `4.2`) teaches the model the actual distance between power levels, rather than treating them as isolated categories.
- **Huber Loss (Smooth L1):** Punishes small errors normally but scales linearly for massive outliers. This protects the model's weights from being destroyed by the "Ego Problem" in user-provided labels.
- **Confidence Weighting:** Applies a mathematical penalty to auto-labeled heuristic decks, forcing the model to trust verified tournament lists over automated checklist assignments.

## 4. Current Status

Sanity checks confirm the embedding space is completely stable and highly accurate. The "Mechanics vs. Meta" tuning knob perfectly balances the vectors: Tutors cluster strictly with Tutors, and fast mana with fast mana. The latent space is locked, and the dataset is ready to be passed into the **Quantity-Aware Transformer Regressor** for final training and ablation testing.
