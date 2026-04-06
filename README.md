# MTG Deck Evaluator

Streamlit app for scoring EDH decklists with a trained SetTransformer model.

## What This Repo Is Now

This project is now focused on inference only:

- Load pre-trained model artifacts
- Paste commander and main deck cards into the UI
- Get one output: calibrated score

Legacy research and data collection scripts are still in the repository, but the main user-facing workflow is the Streamlit app.

## Quick Start

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit in your terminal.

## Using the App

The app has two input boxes:

- Commander Cards
- Main Deck Cards

Supported line formats:

- `1 Card Name`
- `1x Card Name`
- `Card Name` (defaults to quantity 1)

Click Score Deck to run inference. The UI displays only the calibrated score.

## Required Model Files

By default, the app expects these files in the repository root:

- `set_transformer_master_run.pt`
- `896dim_oracle_embeddings.pt`
- `set_transformer_master_run_isotonic_calibrator.joblib`

You can override these paths in the sidebar model settings.

## CLI Scoring (Optional)

If you want scoring without the UI:

```bash
python score_decklist.py --decklist-file path/to/deck.txt --json
```

You can also pipe deck text via stdin.

## Project Layout (Current Focus)

- `streamlit_app.py`: Streamlit UI for deck scoring
- `score_decklist.py`: SetTransformer inference pipeline and parser
- `requirements.txt`: Python dependencies
- `old/`, `helpers/`, `models/`: legacy project assets and experiments

## Data and Credit

Deck data for this work came from Moxfield and MTGTop8.

Huge shoutout to Moxfield for providing deck data access that made this project possible.

## Author

Kevin Peng
