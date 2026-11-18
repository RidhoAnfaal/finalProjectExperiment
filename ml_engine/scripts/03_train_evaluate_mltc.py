#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 2 DATA EXPLORATORY ANALYSIS (EDA) / CHECK GROUND TRUTH 
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT /"data"/"raw"/"jtkl_articles_archive.csv" 

# Cek file csv hasil dari harvesting data jurnal
df=pd.read_csv(RAW_CSV_PATH)
print(f"Size Dataset (Baris, Kolom): {df.shape}")
print(f"Artikel Duplikat: {df['article_url'].duplicated().sum()}")
print("\nJumlah Data Kosong (Missing Values):")
print(df[["title", "abstract", "keywords"]].isna().sum())

# Cek file csv ground truth (labeled data)
word_counts = df["abstract"].str.split().str.len()
print(f'\nStatistik Jumlah Kata Abstrak | Avg : {word_counts.mean():.1f} | Min : {word_counts.min()} | Max : {word_counts.max()}')
LABELED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "jtkl_labeled.csv"
df_labeled = pd.read_csv(LABELED_CSV_PATH)

# Cek persebaran class 
LABEL_COLS = [col for col in df_labeled.columns if col.startswith("label_") and col != "label_count"]
print("\nSebaran Jumlah Artikel per Label (Aim & Scope JTKL):\n")
print(df_labeled[LABEL_COLS].sum())
print(f"\nAverage Label per Artikel : {df_labeled[LABEL_COLS].sum(axis=1).mean():.2f}")

#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 3 DATA PREPROCESSING 
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

# Stopwords ID & EN
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | set(ENGLISH_STOP_WORDS)
print(f"\nTotal Stopwords (ID + EN) : {len(STOPWORDS)} kata")

# Method untuk cleaning text
def clean_text(text:str) -> str :
    text = str(text).lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    filtered = [w for w in tokens if w not in STOPWORDS]
    return " ".join(filtered)

df_labeled["raw_text"] = df_labeled["title"] + " " + df_labeled["title"] + " " + df_labeled["keywords"] + " " + df_labeled["keywords"] + " " + df_labeled["abstract"] 
df_labeled["clean_no_stem"] = df_labeled["raw_text"].apply(clean_text)
# cek hasil cleaning data (lower case, del num, del stopwords)
print("\n[Before Cleaning]:", df_labeled["raw_text"].iloc[0][:120])
print("[After Cleaning]:", df_labeled["clean_no_stem"].iloc[0][:120])

# Stemming Sastrawi VS Non Stemming
import json
STEM_CACHE = json.loads((PROJECT_ROOT / 'data' / "processed" / "sastrawi_stem_cache.json").read_text(encoding="utf-8"))
df_labeled['clean_with_stem'] = df_labeled['clean_no_stem'].apply(lambda txt: ' '.join(STEM_CACHE.get(w, w) for w in txt.split()))
# comparision between unstemmed & stemmed data ()
print("\n[Contoh text ID without stem]:", df_labeled["clean_no_stem"].iloc[68][:110])
print("[Contoh text ID with stem Sastrawi]:", df_labeled["clean_with_stem"].iloc[68][:110])

# Visualitation
from collections import Counter
import matplotlib.pyplot as plt

top_words = Counter(" ".join(df_labeled["clean_no_stem"]).split()).most_common(15)
plt.figure(figsize=(9, 5))
plt.barh([w for w, _ in top_words][::-1], [c for _, c in top_words][::-1], color="teal")
plt.title("Top 15 Kata Teknis Setelah Preprocessing (Stopword Removal)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "processed" / "chart_01_top_words.png", dpi=150); plt.close()

#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 4 MODEL TRAINING FASTTEXT & SVM
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

# Fasttext Words Splitter
def extract_subwords(text: str, min_n: int = 3, max_n: int = 6) -> list[str]:
    subwords = []
    for w in text.split():
        b = f"<{w}>"
        subwords.extend([b] + [b[i:i+n] for n in range(min_n, min(max_n+1, len(b)+1)) for i in range(len(b)-n+1)])
    return subwords
# test extract subwords
print("\nTest Subword : ", extract_subwords("computer vision and image processing"))

# Function vector fasttext
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

def build_fasttext_vectors(train_docs, val_docs, dim: int = 100):
    sub_vec = TfidfVectorizer(analyzer=extract_subwords, sublinear_tf=True, min_df=2, max_features=12000)
    X_tr_sub, X_va_sub = sub_vec.fit_transform(train_docs), sub_vec.transform(val_docs)
    svd, norm = TruncatedSVD(n_components=dim, random_state=42), Normalizer()
    return np.hstack([norm.fit_transform(svd.fit_transform(X_tr_sub)), X_tr_sub.toarray()]), np.hstack([norm.transform(svd.transform(X_va_sub)), X_va_sub.toarray()])

# 5 fold cross validation & SVM
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

Y = df_labeled[LABEL_COLS].values
oof_preds = np.zeros_like(Y)

# Training 5 fold cross validation
corpus = df_labeled['clean_with_stem'].tolist()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(corpus):
    X_tr, X_va = build_fasttext_vectors([corpus[i] for i in train_idx], [corpus[i] for i in val_idx])
    oof_preds[val_idx] = OneVsRestClassifier(LinearSVC(C=0.6, class_weight='balanced', random_state=42)).fit(X_tr, Y[train_idx]).predict(X_va)

#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 5 MODEL EVALUATION 
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, classification_report

print(f"\nHasil Evaluasi Model FastText + SVM (5 fold cross validation) :")
print(f"Macro-F1 Score    : {f1_score(Y, oof_preds, average='macro')*100:.2f}%")
print(f"Weighted-F1 Score : {f1_score(Y, oof_preds, average='weighted')*100:.2f}%")
print(f"Hamming Loss      : {hamming_loss(Y, oof_preds):.4f}")

print('\nDetail Performance per Label Aim & Scope:')
print(classification_report(Y, oof_preds, target_names=LABEL_COLS, zero_division=0))

#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 6 Joblib
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# import joblib

# sub_vec = TfidfVectorizer(analyzer=extract_subwords, sublinear_tf=True, min_df=2, max_features=12000).fit(corpus)
# X_sub = sub_vec.transform(corpus)
# svd, norm = TruncatedSVD(n_components=100, random_state=42).fit(X_sub), Normalizer()
# clf_final = OneVsRestClassifier(LinearSVC(C=0.6, class_weight="balanced", random_state=42)).fit(np.hstack([norm.fit_transform(svd.transform(X_sub)), X_sub.toarray()]), Y)