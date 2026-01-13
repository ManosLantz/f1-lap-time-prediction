
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

if __name__ == "__main__":
    df, _ = load_and_prep()
    print("Seasons found:", df['Season'].unique())
    print("Counts:\n", df['Season'].value_counts())
