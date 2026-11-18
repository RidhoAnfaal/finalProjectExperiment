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


#########################################################
# 3 DATA PREPROCESSING 
#########################################################

# Stopwords ID & EN
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | set(ENGLISH_STOP_WORDS)
print(f"\nTotal Kamus Stopwords (ID + EN) : {len(STOPWORDS)} kata")

# Cleaning Text
def clean_text(text:str) -> str :
    text = str(text).lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    filtered = [w for w in tokens if w not in STOPWORDS]
    return " ".join(filtered)
    
df_labeled["raw_text"] = df_labeled["title"] + " " + df_labeled["title"] + " " + df_labeled["keywords"] + " " + df_labeled["keywords"] + " " + df_labeled["abstract"] 
df_labeled["clean_no_stem"] = df_labeled["raw_text"].apply(clean_text)
# comparision between unclean and cleaned data (lower case, del num, del stopwords)
print("\n[SEBELUM CLEAN]:", df_labeled["raw_text"].iloc[0][:120])
print("[SESUDAH CLEAN]:", df_labeled["clean_no_stem"].iloc[0][:120])

# Stemming Sastrawati VS Non Stemming Sastrawati
import json
STEM_CACHE = json.loads((PROJECT_ROOT / 'data' / "processed" / "sastrawi_stem_cache.json").read_text(encoding="utf-8"))
df_labeled['clean_with_stem'] = df_labeled['clean_no_stem'].apply(lambda txt: ' '.join(STEM_CACHE.get(w, w) for w in txt.split()))
# comparision between unstemmed & stemmed data ()
print("\n[CONTOH ID TANPA STEM ]:", df_labeled["clean_no_stem"].iloc[68][:110])
print("[CONTOH ID DENGAN STEM]:", df_labeled["clean_with_stem"].iloc[68][:110])