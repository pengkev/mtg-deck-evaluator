# Magic: The Gathering Deck Power Level Classifier

## 1. The Initial Approach

Initially, the objective was to map Magic: The Gathering cards into continuous vector spaces using a Word2Vec embedding model, aggregate those card vectors into 100-card decks, and train a classifier to predict 5 distinct power levels (Brackets 1–5).

The classifier was originally built using a **Deep Sets architecture**. Because Deep Sets can be difficult to interpret and struggle with contextual feature extraction, a **Single-Head Attention-Based pooling layer** was introduced. While this allowed for attention-weight visualization (showing which cards the model thought were important), the overall classification performance flatlined at roughly ~20–21%—effectively random guessing.

---

## 2. Data Pipeline & Labeling Discoveries

A deeper diagnostic into the ~20% mode collapse revealed two critical flaws in the **data pipeline**, rather than the architecture:

* **The `<UNK>` Avalanche:** The raw scraped decks used inconsistent naming conventions for split cards, Adventure cards, and MDFCs (e.g., `"Fire // Ice"` vs. `"Fire/Ice"`). This caused a massive vocabulary mismatch where the model was blind to up to 15% of the cards in a deck, mapping them to the `<UNK>` token. Implementing a strict **text-normalization pipeline** across the megacorpus solved this.
* **The Moxfield Heuristic Trap:** Analysis of the labels revealed that Moxfield's auto-bracket system is not a machine learning power evaluator; it is a rigid, rules-based heuristic that checks for specific mechanics (e.g., Mass Land Denial, Fast Mana) and completely ignores synergies and two-card combos. The neural network was failing because it was trying to discover deep mathematical synergies, while the loss function was punishing it for not memorizing Moxfield's arbitrary checklists.

---

## 3. The 300k Megacorpus & The Latent Space

To build the foundational embeddings, a 512-dimension Word2Vec model was trained on a heavily skewed megacorpus of ~300,000 decks (18 million total cards), intentionally mixing 60-card competitive formats (Legacy, Modern, Vintage) with 100-card Commander decks.

**Key Finding: The Perfect Curriculum**
Using an "infinite" window size (115), the model successfully mapped the mechanical DNA of the game. The competitive 60-card lists forced the model to learn hyper-efficient synergies and staple packages (e.g., *Lion's Eye Diamond* + *Underworld Breach*), while the 115k EDH decks provided the necessary context for Commander-exclusive pillars (e.g., *Sol Ring*, *Command Tower*).

However, Word2Vec clusters cards by "archetype packages" rather than functional equivalence. It inadvertently places ubiquitous mana-fixing in the same latent space as high-power win conditions. Therefore, a robust self-attention mechanism is strictly required downstream so the classifier can dynamically mute high-frequency staples and amplify actual power-level outliers.

---

## 4. The Classification Confusion Matrix & "The Ego Problem"

Testing the Deep Sets architecture on a 165k labeled dataset of Brackets 1–5 yielded a highly revealing confusion matrix.  While the model successfully learned to separate low-power battlecruisers from high-power cEDH shells (forming a strong diagonal), it exposed severe issues with the ground truth data:

* **The B1/B2 Blur:** The line between unmodified precons (B1) and slightly upgraded precons (B2) is mathematically microscopic. The model frequently misclassified B1s as B2s the moment it detected a few generic staples.
* **The B4/B5 Squeeze:** High-Power Casual (B4) and fringe cEDH (B5) share roughly 90% of the same structural DNA (fast mana, efficient interaction). Deep Sets struggled to separate them because it simply aggregates card weights without understanding *how* the cards interact.
* **The Human Ego Problem:** The model frequently classified user-labeled "Bracket 5" decks as Bracket 2 or Bracket 3. Manual inspection confirmed the neural network was actually correct; users frequently miscalibrate their own deck's power level, labeling average mid-power decks as cEDH out of bias or lack of format knowledge.

---

## 5. Current Direction: Transformer Encoders & Ordinal Regression

To solve the B4/B5 squeeze and protect against human labeling bias, the architecture is pivoting entirely in two distinct ways:

### Architectural Upgrade: Deep Sets to Transformer

Deep Sets treat cards in a vacuum. A **Transformer Encoder**  allows the cards to dynamically update their own mathematical representations via Self-Attention before the final power level is predicted. This solves the "Lion's Eye Diamond Problem"—the model can now recognize that an LED in a deck with *Brain Freeze* is a lethal Bracket 5 combo, while an LED in a deck with zero graveyard synergy is effectively harmless.

### Objective Upgrade: Classification to Ordinal Regression

Power levels are a continuous spectrum. Standard Cross-Entropy loss punishes a network equally whether it guesses a 1 for a 5-level deck, or a 4 for a 5-level deck. By switching to a single continuous output (e.g., predicting `4.2`), the model learns the *distance* between power levels.

* **Outlier Protection via Huber Loss:** The model utilizes **Huber Loss (Smooth L1)**. MSE quadratically punishes large errors, which is dangerous in a dataset heavily polluted by the "Ego Problem" (users labeling a B2 deck as a B5). Huber loss acts like MSE for small errors to aid smooth convergence, but scales linearly for massive outliers, protecting the model's weights from highly inaccurate human labels.
* **Confidence Weighting:** The dataset labels are drawn from user assignments, Moxfield auto-labels, and verified tournament cEDH lists. A custom PyTorch dataset class applies a mathematical penalty to the loss function for auto-labeled decks, forcing the model to trust verified tournament lists over heuristics.
