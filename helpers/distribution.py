import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load your index
df = pd.read_csv("../data/mtgtop8-general/all_decks_index.csv")

# Count decks by format
print(df['format'].value_counts())