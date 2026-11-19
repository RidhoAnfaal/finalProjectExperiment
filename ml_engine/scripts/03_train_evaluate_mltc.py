#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 2 DATA EXPLORATORY ANALYSIS (EDA) / CHECK GROUND TRUTH 
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
from matplotlib.collections import PathCollection
from typing import Callable
from matplotlib.lines import lineStyles
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
print("\nSebaran Jumlah Artikel per Label (Aim & Scope JTKL):")
print(df_labeled[LABEL_COLS].sum())
print(f"\nAverage Label per Artikel : {df_labeled[LABEL_COLS].sum(axis=1).mean():.2f}")

# visualisasi persebaran
import matplotlib.pyplot as plt

label_counts = df_labeled[LABEL_COLS].sum().sort_values()
plt.figure(figsize=(9, 5)); plt.grid(True, linestyle="--", alpha=0.5)
sc = plt.scatter(label_counts.values, label_counts.index, s=label_counts.values*12, c=label_counts.values, cmap="viridis", edgecolors="k", zorder=3)
for x, y in zip(label_counts.values, label_counts.index): plt.text(x + 1.5, y, f"{x} artikel", va="center", fontsize=9)
plt.xlim(0, 100); plt.colorbar(sc, label="Jumlah Artikel"); plt.xlabel("Jumlah Artikel"); plt.title("Sebaran Jumlah Artikel per Label (Aim & Scope JTKL)"); plt.tight_layout(); plt.savefig(PROJECT_ROOT / "data" / "processed" / "chart_00_eda_scatter.png", dpi=150); plt.close()

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

# Function hybrid vectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

def make_hybrid_vectorizer():
    return FeatureUnion([("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=5000)), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=10000))]) 

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
    vec = make_hybrid_vectorizer()
    X_tr, X_va = vec.fit_transform([corpus[i] for i in train_idx]), vec.transform([corpus[i] for i in val_idx])    
    oof_preds[val_idx] = (OneVsRestClassifier(LinearSVC(C=0.6, class_weight="balanced", random_state=42)).fit(X_tr, Y[train_idx]).decision_function(X_va) >= -0.15).astype(int)
    
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

# Visualisasi SVM Decision Boundary & Margin (Studi Kasus Label 1: Energy vs Non-Energy)
from sklearn.decomposition import TruncatedSVD
X_all = make_hybrid_vectorizer().fit_transform(corpus); y_energy = Y[:, 0]
X_2d = np.column_stack([LinearSVC(C=0.6, class_weight="balanced", random_state=42).fit(X_all, y_energy).decision_function(X_all), TruncatedSVD(n_components=2, random_state=42).fit_transform(X_all)[:, 1] * 3])

xx, yy = np.meshgrid(np.linspace(X_2d[:,0].min()-0.3, X_2d[:,0].max()+0.3, 200), np.linspace(X_2d[:,1].min()-0.3, X_2d[:,1].max()+0.3, 200))
Z = xx
plt.figure(figsize=(8, 5)); plt.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=["#3498db", "#e74c3c"], alpha=0.18)
plt.contour(xx, yy, Z, levels=[-0.5, 0, 0.5], colors="k", linestyles=["--", "-", "--"], linewidths=[1.5, 2.5, 1.5])
plt.scatter(X_2d[y_energy==0,0], X_2d[y_energy==0,1], c="#2980b9", label="Non-Energy (0)", edgecolors="k"); plt.scatter(X_2d[y_energy==1,0], X_2d[y_energy==1,1], c="#c0392b", label="Energy (1)", edgecolors="k"); plt.legend(); plt.title("SVM Decision Boundary (Garis Tegas) & Margin (Putus-putus) - Topik Energy"); plt.tight_layout(); plt.savefig(PROJECT_ROOT / "data" / "processed" / "chart_02_svm_decision_boundary.png", dpi=150); plt.close()

#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# 6 Joblib
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
import joblib

vec_final = make_hybrid_vectorizer().fit(corpus)
clf_final = OneVsRestClassifier(LinearSVC(C=0.6, class_weight="balanced", random_state=42)).fit(vec_final.transform(corpus), Y)
MODEL_PATH = PROJECT_ROOT / "models" / "jtkl_model_v1.joblib"
joblib.dump({"vec": vec_final, "clf": clf_final, "labels": LABEL_COLS}, MODEL_PATH)

sample_clean = " ".join(STEM_CACHE.get(w, w) for w in clean_text("Comparative Gasification Performance of Cassava Stems Bagasse and Sengon Sawdust").split())
pred = clf_final.predict(vec_final.transform([sample_clean]))[0]
print(f"\nTest hasil prediksi artikel ({sample_clean}):", [LABEL_COLS[i] for i, val in enumerate(pred) if val == 1])