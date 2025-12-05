# utils.py

import pandas as pd

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
