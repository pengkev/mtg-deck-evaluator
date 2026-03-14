# Magic: The Gathering Deck Power Level Classifier

## 1. The Initial Approach

Initially, the objective was to map Magic: The Gathering cards into continuous vector spaces
using a Word2Vec embedding model, aggregate those card vectors into 100-card decks, and train
a classifier to predict 5 distinct power levels (Brackets 1–5).

The classifier was originally built using a **Deep Sets architecture**. Because Deep Sets can
be difficult to interpret and struggle with contextual feature extraction, a **Single-Head
Attention-Based pooling layer** was introduced. While this allowed for attention-weight visualization
(showing which cards the model thought were important), the overall classification performance
flatlined at roughly ~20–21%—effectively random guessing.

---

## 2. Data Pipeline & Labeling Discoveries

A deeper diagnostic into the ~20% mode collapse revealed two critical flaws in the **data
pipeline**, rather than the architecture:

1. **The `<UNK>` Avalanche:** The raw scraped decks used inconsistent naming conventions for
   split cards, Adventure cards, and MDFCs (e.g., `"Fire // Ice"` vs. `"Fire/Ice"`). This
   caused a massive vocabulary mismatch where the model was blind to up to 15% of the cards
   in a deck, mapping them to the `<UNK>` token. Implementing a strict
   **text-normalization pipeline** across the megacorpus solved this.

2. **The Moxfield Heuristic Trap:** Analysis of the labels revealed that Moxfield's
   auto-bracket system is not a machine learning power evaluator; it is a rigid, rules-based
   heuristic that checks for specific mechanics (e.g., Mass Land Denial, Fast Mana) and
   completely ignores synergies and two-card combos. The neural network was failing because it
   was trying to discover deep mathematical synergies, while the loss function was punishing it
   for not memorizing Moxfield's arbitrary `if/else` checklists.

---

## 3. The Binary Classification Breakthrough

To verify that the underlying Gensim embeddings and Attention architecture actually worked,
the problem was simplified. Brackets 1–4 were collapsed into a single **"Casual"** class, and
Bracket 5 (supplemented with tournament-verified MTGTop8 cEDH lists) formed a
**"Competitive"** class.

This Binary Classification model achieved highly promising results (**~85% accuracy**).
Furthermore, extracting the attention weights proved the model successfully learned the
semantic difference between cEDH staple packages (fast mana, free interaction) and casual
synergistic pieces.

**Key Finding: Embedding Behavior & The "Staple Trap"**
Word2Vec embeddings (trained with a window size of 115) cluster cards by "archetype packages"
rather than functional equivalence. While this is fantastic for identifying high-power shells,
it inadvertently places ubiquitous mana-fixing (fetches, _Command Tower_) in the same latent
space as win-conditions (_Underworld Breach_). Therefore, a robust Attention mechanism is
strictly required so the classifier can dynamically mute high-frequency staples and amplify
the actual power-level outliers.

---

## 4. Current Direction: Multi-Head Attention & Ordinal Regression

With the data pipeline cleaned and the embeddings proven, the focus has returned to the 1–5
scale. A massive scrape of user-labeled Moxfield decks is currently underway. However, moving
forward, the architecture is pivoting from **Categorical Classification** to
**Ordinal Regression**.

- **Why Regression?** Power levels are a continuous spectrum. Standard Cross-Entropy loss
  punishes a network equally whether it guesses a 1 for a 5-level deck, or a 4 for a 5-level
  deck. By switching to a single continuous output (e.g., predicting `4.2`), the model learns
  the _distance_ between power levels.

- **Outlier Protection via Huber Loss:** Instead of standard Mean Squared Error (MSE), the
  model utilizes **Huber Loss (Smooth L1)**. MSE quadratically punishes large errors, which is
  dangerous in a dataset prone to human ego and miscalibration (e.g., a user labeling a Bracket 5
  deck as a 1). Huber loss acts like MSE for small errors to aid smooth convergence, but scales
  linearly for massive outliers, protecting the model's weights from highly inaccurate human labels.

- **Handling Label Uncertainty (Empirical Sample Weighting):** The new dataset contains labels on a
  strict 1–5 integer scale drawn from two sources: human users and the Moxfield auto-labeler.
  Because human labels are deemed higher quality, they act as the primary Ground Truth (`Weight = 1.0`).
  Instead of applying an arbitrary penalty to auto-labeled decks, the confidence weight is derived
  empirically. By isolating the subset of decks containing _both_ a user label and an auto-label, we
  calculate the Mean Absolute Error (MAE) between the two. This average distance is then used to
  mathematically scale down the Huber loss for auto-only decks, penalizing them in exact proportion
  to their historical divergence from human consensus.
