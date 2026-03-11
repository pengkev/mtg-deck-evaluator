import pandas as pd
for bracket in['1','2','3','4','5']:
    # Load your index
    df = pd.read_csv(f"../data/moxfield-edh-bracket-{bracket}/moxfield_log.csv")

    # Count decks by format
    print(len(df['id']))