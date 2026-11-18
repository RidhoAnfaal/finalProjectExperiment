#########################################################
# 2 DATA EXPLORATORY ANALYSIS (EDA) / CHECK GROUND TRUTH 
#########################################################
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT /"data"/"raw"/"jtkl_articles_archive.csv" 

# Check File (jtkl labeled)
df=pd.read_csv(RAW_CSV_PATH)
print(f"Ukuran Dataset (Baris, Kolom): {df.shape}")
print(f"Artikel Duplikat: {df['article_url'].duplicated().sum()}")
print("\nJumlah Data Kosong (Missing Values):\n", df[["title", "abstract", "keywords"]].isna().sum())

# Check Text Distributions (jtkl labeled)
word_counts = df["abstract"].str.split().str.len()
print(f'\nStatistik Jumlah Kata Abstrak | Avg : {word_counts.mean():.1f} | Min : {word_counts.min()} | Max : {word_counts.max()}')
LABELED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "jtkl_labeled.csv"
df_labeled = pd.read_csv(LABELED_CSV_PATH)

# Check Classes (jtkl labeled)
LABEL_COLS = [col for col in df_labeled.columns if col.startswith("label_") and col != "label_count"]
print("\nSebaran Jumlah Artikel per Label (Aim & Scope JTKL):\n", df_labeled[LABEL_COLS].sum())
print(f"\nRata-rata Label per Artikel (Label Cardinality): {df_labeled[LABEL_COLS].sum(axis=1).mean():.2f}")

# 3 DATA PREPROCESSING 

