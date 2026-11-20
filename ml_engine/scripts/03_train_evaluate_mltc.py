"""
Multi-Label Text Classification menggunakan LinearSVC dengan Hybrid Word & Subword Character N-Gram TF-IDF
Jurnal Teknik Kimia dan Lingkungan (JTKL) Polinema
Script di sini merupakan experiment untuk memastikan metode/pipeline yang saya telah pikirkan dan rencanakan terkonfirmasi berjalan dengan baik dan hasil yang memuaskan

"""

from pathlib import Path
import re
import json
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report
)

#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
# 1. SETUP PATH & LOAD DATA
#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "jtkl_articles_archive.csv"
LABELED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "jtkl_labeled.csv"
STEM_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "sastrawi_stem_cache.json"

#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
# 2. EXPLORATORY DATA ANALYSIS (EDA)
#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0

# Cek raw data hasil harvesting
df_raw = pd.read_csv(RAW_CSV_PATH)
print(f"Size dataset raw (baris, kolom): {df_raw.shape}")
print(f"Artikel duplikat: {df_raw['article_url'].duplicated().sum()}")
print("\nJumlah missing values:")
print(df_raw[["title", "abstract", "keywords"]].isna().sum())

# Cek statistik panjang abstrak
word_counts = df_raw["abstract"].dropna().str.split().str.len()
print(f"\nStatistik jumlah kata abstrak | Avg: {word_counts.mean():.1f} | Min: {word_counts.min()} | Max: {word_counts.max()}")

# Load data ground truth berlabel
df_labeled = pd.read_csv(LABELED_CSV_PATH)
LABEL_COLS = [col for col in df_labeled.columns if col.startswith("label_") and col != "label_count"]

print("\nSebaran jumlah artikel per label (Aim & Scope JTKL):")
print(df_labeled[LABEL_COLS].sum())
print(f"\nAverage label per artikel: {df_labeled[LABEL_COLS].sum(axis=1).mean():.2f}")

# Visualisasi sebaran label 
label_counts = df_labeled[LABEL_COLS].sum().sort_values()
plt.figure(figsize=(9, 5))
plt.grid(True, linestyle="--", alpha=0.5)

sc = plt.scatter(
    label_counts.values,
    label_counts.index,
    s=label_counts.values * 12,
    c=label_counts.values,
    cmap="viridis",
    edgecolors="k",
    zorder=3
)

for x, y in zip(label_counts.values, label_counts.index):
    plt.text(x + 1.5, y, f"{x} artikel", va="center", fontsize=9)

plt.xlim(0, 100)
plt.colorbar(sc, label="Jumlah Artikel")
plt.xlabel("Jumlah Artikel")
plt.title("Sebaran Jumlah Artikel per Label (Aim & Scope JTKL)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "processed" / "chart_00_eda_scatter.png", dpi=150)
plt.close()

#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
# 3. TEXT PREPROCESSING & STEMMING
#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0

# Gabungan Stopwords Bahasa Indonesia (Sastrawi) & Bahasa Inggris (Scikit-Learn)
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | set(ENGLISH_STOP_WORDS)
print(f"\nTotal Stopwords (ID + EN): {len(STOPWORDS)} kata")

def clean_text(text: str) -> str:
    text = str(text).lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    filtered = [w for w in tokens if w not in STOPWORDS]
    return " ".join(filtered)

# Pembobotan domain metadata
df_labeled["raw_text"] = (
    df_labeled["title"] + " " + df_labeled["title"] + " " +
    df_labeled["keywords"].fillna("") + " " + df_labeled["keywords"].fillna("") + " " +
    df_labeled["abstract"].fillna("")
)
df_labeled["clean_no_stem"] = df_labeled["raw_text"].apply(clean_text)

# cache kamus hasil stemming Sastrawi
STEM_CACHE = json.loads(STEM_CACHE_PATH.read_text(encoding="utf-8"))

def apply_stemming(text: str) -> str:
    return " ".join(STEM_CACHE.get(word, word) for word in text.split())

df_labeled["clean_with_stem"] = df_labeled["clean_no_stem"].apply(apply_stemming)

print("\n[Before cleaning]:", df_labeled["raw_text"].iloc[0][:120])
print("[After cleaning] :", df_labeled["clean_no_stem"].iloc[0][:120])
print("\n[Unstemmed]      :", df_labeled["clean_no_stem"].iloc[68][:110])
print("[Stemmed sastrawi]:", df_labeled["clean_with_stem"].iloc[68][:110])

# Visualisasi top 15 Kata Kunci Teknis
top_words = Counter(" ".join(df_labeled["clean_no_stem"]).split()).most_common(15)
plt.figure(figsize=(9, 5))
plt.barh(
    [w for w, _ in top_words][::-1],
    [c for _, c in top_words][::-1],
    color="teal"
)
plt.title("Top 15 Kata Teknis Setelah Preprocessing (Stopword Removal)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "processed" / "chart_01_top_words.png", dpi=150)
plt.close()

#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
# 4. FEATURE EXTRACTION & MODEL TRAINING (SVM 5-FOLD CV)
#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
def make_hybrid_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                max_features=5000
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=2,
                max_features=10000
            )
        )
    ])

Y = df_labeled[LABEL_COLS].values
oof_preds = np.zeros_like(Y)
corpus = df_labeled["clean_with_stem"].tolist()

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(corpus), 1):
    vec = make_hybrid_vectorizer()
    X_tr = vec.fit_transform([corpus[i] for i in train_idx])
    X_va = vec.transform([corpus[i] for i in val_idx])

    clf = OneVsRestClassifier(
        LinearSVC(C=0.6, class_weight="balanced", random_state=42)
    )
    clf.fit(X_tr, Y[train_idx])

    # Out-of-Fold dengan threshold tuning = -0.15
    scores = clf.decision_function(X_va)
    oof_preds[val_idx] = (scores >= -0.15).astype(int)

#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
# 5. MODEL EVALUATION & VISUALIZATION
#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
print("\nHasil Evaluasi Model Hybrid TF-IDF + LinearSVC (5-Fold Cross Validation):")
print(f"Macro-F1 Score    : {f1_score(Y, oof_preds, average='macro') * 100:.2f}%")
print(f"Weighted-F1 Score : {f1_score(Y, oof_preds, average='weighted') * 100:.2f}%")
print(f"Hamming Loss      : {hamming_loss(Y, oof_preds):.4f}")

print("\nDetail Performance per Label Aim & Scope:")
print(classification_report(Y, oof_preds, target_names=LABEL_COLS, zero_division=0))

# Visualisasi SVM (Label Energy)
X_all = make_hybrid_vectorizer().fit_transform(corpus)
y_energy = Y[:, 0]

# Margin decision score + SVD component
svm_energy = LinearSVC(C=0.6, class_weight="balanced", random_state=42).fit(X_all, y_energy)
svd_comp = TruncatedSVD(n_components=2, random_state=42).fit_transform(X_all)[:, 1] * 3
X_2d = np.column_stack([svm_energy.decision_function(X_all), svd_comp])

xx, yy = np.meshgrid(
    np.linspace(X_2d[:, 0].min() - 0.3, X_2d[:, 0].max() + 0.3, 200),
    np.linspace(X_2d[:, 1].min() - 0.3, X_2d[:, 1].max() + 0.3, 200)
)
Z = xx

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=["#3498db", "#e74c3c"], alpha=0.18)
plt.contour(xx, yy, Z, levels=[-0.5, 0, 0.5], colors="k", linestyles=["--", "-", "--"], linewidths=[1.5, 2.5, 1.5])
plt.scatter(
    X_2d[y_energy == 0, 0],
    X_2d[y_energy == 0, 1],
    c="#2980b9",
    label="Non-Energy (0)",
    edgecolors="k"
)
plt.scatter(
    X_2d[y_energy == 1, 0],
    X_2d[y_energy == 1, 1],
    c="#c0392b",
    label="Energy (1)",
    edgecolors="k"
)
plt.legend()
plt.title("SVM Decision Boundary & Margin - Topik/Label Energy")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "processed" / "chart_02_svm_decision_boundary.png", dpi=150)
plt.close()

#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0
# 6. STORE MODEL
#o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0o0

# Latih model final pada seluruh corpus untuk disimpan ke bundle
vec_final = make_hybrid_vectorizer().fit(corpus)
clf_final = OneVsRestClassifier(
    LinearSVC(C=0.6, class_weight="balanced", random_state=42)
).fit(vec_final.transform(corpus), Y)

MODEL_PATH = PROJECT_ROOT / "models" / "jtkl_model_v1.joblib"
joblib.dump(
    {
        "vec": vec_final,
        "clf": clf_final,
        "labels": LABEL_COLS
    },
    MODEL_PATH
)

# Testing artikel sampel
sample_title = "Comparative Gasification Performance of Cassava Stems Bagasse and Sengon Sawdust"
sample_clean = apply_stemming(clean_text(sample_title))
sample_vec = vec_final.transform([sample_clean])
sample_scores = clf_final.decision_function(sample_vec)[0]
predicted_labels = [LABEL_COLS[i] for i, score in enumerate(sample_scores) if score >= -0.15]

print(f"\nUji Coba Inferensi:")
print(f"Judul Input   : {sample_title}")
print(f"Hasil Prediksi: {predicted_labels}")
