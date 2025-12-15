import pandas as pd
import numpy as np

DATASET_PATH = "./2021_german_doctor_reviews.csv"

def load_data(sample_frac=0.3, min_words=10):
    df = pd.read_csv(DATASET_PATH).dropna()
    
    df['sentiment'] = df['rating'].apply(lambda r: (
        "Positive" if float(r) <= 2 else
        "Mixed" if float(r) <= 4 else "Negative"
    ) if str(r).replace('.', '', 1).isdigit() else "Unknown")
    
    return (
        df[df.sentiment != 'Unknown']
          .groupby('sentiment', group_keys=False)
          .apply(lambda x: x.sample(frac=sample_frac, random_state=0))
          .loc[df.comment.str.split().str.len() > min_words]
    )
