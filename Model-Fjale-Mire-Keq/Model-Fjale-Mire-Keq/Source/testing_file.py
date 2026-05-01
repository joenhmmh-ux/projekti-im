import pickle
import json
import csv
import io
import pandas as pd
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent

df = pd.read_csv( base_dir / "Data" / "train_data_sq.csv")
print(df.columns.tolist())