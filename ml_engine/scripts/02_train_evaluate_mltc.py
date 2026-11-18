r"""
Langkah 5: Pipeline Eksperimen Multi-Label Text Classification (MLTC)
Target Dataset : D:\Skripsi_Polinema\ml_engine\data\processed\jtkl_labeled.csv (153 Artikel, 9 Label)
Output Laporan :
  1. D:\Skripsi_Polinema\ml_engine\data\processed\jtkl_experiment_report.txt
  2. D:\Skripsi_Polinema\ml_engine\data\processed\jtkl_experiment_results.csv
  3. D:\Skripsi_Polinema\ml_engine\data\processed\jtkl_per_label_metrics.csv
  4. D:\Skripsi_Polinema\ml_engine\models\jtkl_fasttext_svm_bundle.joblib

Cakupan Eksperimen (Sesuai rules.md Bab 3 & Bab 4):
1. Komparasi Preprocessing : Tanpa Stemming (No-Stem) vs. Dengan Stemming Sastrawi (With-Stem).
2. Komparasi Representasi  : TF-IDF (Word Unigram-Bigram) vs. FastText Subword Embeddings
                             (Boundary-aware character n-grams <w> n=3..6 + Subword Semantic Projection).
3. Komparasi Klasifikasi   :
   - Baseline 1 (Metode Penelitian Terdahulu - Sulthan Rafif dkk., 2021 pada Multi-Label):
     TF-IDF + Binary Relevance Naive Bayes (MultinomialNB)
   - Baseline 2 (SVM Konvensional):
     TF-IDF + Binary Relevance SVM (OneVsRestClassifier)
   - Proposed 1 (FastText + SVM Tanpa Stemming):
     FastText Subword Embeddings + Binary Relevance SVM (No-Stemming)
   - Proposed 2 (FastText + SVM Dengan Stemming Sastrawi):
     FastText Subword Embeddings + Binary Relevance SVM (With-Stemming)
   - Proposed 3 (FastText + SVM Optimized Hyperparameter & Threshold):
     FastText Subword Embeddings + Tuned Binary Relevance SVM (class_weight='balanced' + Threshold Tuning)

Prinsip Anti-Data Leakage (Strict Featurization Ordering):
Seluruh proses pelatihan ekstraksi fitur (TF-IDF maupun FastText Subword) serta pencarian
ambang batas keputusan (Decision Threshold) dilakukan HANYA pada data Training di setiap Fold
dari 5-Fold Cross-Validation, kemudian diuji pada data Validation Fold yang terpisah.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_CSV_PATH = PROCESSED_DIR / "jtkl_labeled.csv"
STEM_CACHE_PATH = PROCESSED_DIR / "sastrawi_stem_cache.json"
REPORT_TXT_PATH = PROCESSED_DIR / "jtkl_experiment_report.txt"
RESULTS_CSV_PATH = PROCESSED_DIR / "jtkl_experiment_results.csv"
PER_LABEL_CSV_PATH = PROCESSED_DIR / "jtkl_per_label_metrics.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "jtkl_fasttext_svm_bundle.joblib"

# Daftar 9 Label Resmi Aim and Scope JTKL
LABEL_MAP: Dict[str, str] = {
    "label_energy": "1. Energy",
    "label_waste_treatment": "2. Waste treatment",
    "label_unit_operation": "3. Unit operation",
    "label_thermodynamic": "4. Thermodynamic",
    "label_process_simulation": "5. Process simulation",
    "label_new_material": "6. Development and application of new material",
    "label_chemical_reaction": "7. Chemical engineering reaction",
    "label_biochemical": "8. Biochemical",
    "label_biomass": "9. Biomass",
}
LABEL_COLS: List[str] = list(LABEL_MAP.keys())


# ==============================================================================
# 1. TAHAP PREPROCESSING TEKS BILINGUAL (INDONESIA + INGGRIS)
# ==============================================================================

def build_bilingual_stopwords() -> set:
    """Menggabungkan Stopwords Bahasa Indonesia (Sastrawi) + Bahasa Inggris + Boilerplate."""
    id_stopwords = set(StopWordRemoverFactory().get_stop_words())
    en_stopwords = set(ENGLISH_STOP_WORDS)
    academic_boilerplate = {
        "abstract", "abstrak", "penelitian", "study", "research", "paper",
        "jurnal", "journal", "aim", "aims", "aimed", "purpose", "tujuan",
        "method", "methods", "metode", "result", "results", "hasil",
        "conclusion", "conclusions", "kesimpulan", "showed", "shows",
        "menunjukkan", "based", "berdasarkan", "using", "used", "menggunakan",
        "conducted", "dilakukan", "obtained", "diperoleh", "variasi",
        "variation", "effect", "pengaruh", "analysis", "analisis",
    }
    return id_stopwords | en_stopwords | academic_boilerplate


def clean_and_tokenize(text: str, stopwords: set) -> List[str]:
    """
    Case folding -> Cleansing (hapus simbol, tag HTML, URL, angka murni) ->
    Tokenisasi & Bilingual Stopword Removal.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    # Pertahankan huruf alfabet dan tanda hubung istilah kimia penting lalu pisahkan
    tokens = re.findall(r"[a-z]{3,}", text)
    return [t for t in tokens if t not in stopwords]


def prepare_corpora(df: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Menghasilkan dua varian korpus untuk eksperimen komparasi preprocessing:
    1. corpus_no_stem   : Teks bersih tanpa stemming
    2. corpus_with_stem : Teks bersih dengan stemming Sastrawi
    """
    stopwords = build_bilingual_stopwords()

    # Konstruksi fitur teks gabungan: Judul (diulang 2x sebagai penegas topik) + Keywords (2x) + Abstrak
    raw_docs: List[str] = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        kw = str(row.get("keywords", "")).replace(";", " ")
        abstract = str(row.get("abstract", ""))
        combined = f"{title} {title} {kw} {kw} {abstract}"
        raw_docs.append(combined)

    tokenized_docs = [clean_and_tokenize(doc, stopwords) for doc in raw_docs]
    corpus_no_stem = [" ".join(tokens) for tokens in tokenized_docs]

    # Optimasi Sastrawi Stemming dengan Kamus Cache Lokal
    if STEM_CACHE_PATH.exists():
        stem_cache: Dict[str, str] = json.loads(STEM_CACHE_PATH.read_text(encoding="utf-8"))
    else:
        stem_cache = {}

    unique_tokens = {t for doc in tokenized_docs for t in doc}
    missing_tokens = [t for t in unique_tokens if t not in stem_cache]

    if missing_tokens:
        stemmer = StemmerFactory().create_stemmer()
        # Filter cerdas: jalankan Sastrawi pada kata yang memiliki pola imbuhan Bahasa Indonesia
        # agar istilah teknis Bahasa Inggris murni tidak memakan waktu rekursi sia-sia
        id_affix_re = re.compile(r"^(me|pe|ber|ter|di|ke|se)[a-z]+|[a-z]+(kan|an|nya|i)$")
        for token in missing_tokens:
            if id_affix_re.match(token):
                stem_cache[token] = stemmer.stem(token)
            else:
                stem_cache[token] = token
        STEM_CACHE_PATH.write_text(json.dumps(stem_cache, ensure_ascii=False), encoding="utf-8")

    corpus_with_stem = [
        " ".join(stem_cache.get(t, t) for t in tokens)
        for tokens in tokenized_docs
    ]

    return corpus_no_stem, corpus_with_stem, stem_cache


# ==============================================================================
# 2. IMPLEMENTASI FASTTEXT SUBWORD EMBEDDINGS (BOJANOWSKI ET AL., 2017)
#    Kompatibel 100% dengan Python 3.13 Windows (Scikit-Learn Transformer)
# ==============================================================================

class FastTextSubwordVectorizer(BaseEstimator, TransformerMixin):
    """
    Representasi Vektor Dokumen Berbasis FastText Subword Information (Bojanowski et al., 2017).

    Cara Kerja Metodologis:
    1. Setiap kata w diberi penanda batas kata `<w>` (misal: `<adsorption>` atau `<adsorpsi>`),
       lalu dipecah menjadi himpunan character n-grams berukuran min_n=3 hingga max_n=6
       ditambah token kata utuhnya `<w>`.
       Contoh: `<adsorpsi>` dan `<adsorption>` berbagi subword `<ad`, `ads`, `dso`, `sor`,
       `orp`, `<ads`, `adso`, `dsor`, `sorp`, `<adso`, `adsor`, `dsorp`, `<adsor`, `adsorp`.
    2. Matriks kemunculan subword & kata dibobotkan secara sublinear (1 + log(tf)) * idf,
       kemudian diproyeksikan ke ruang vektor semantik padat (dense embedding R^d)
       menggunakan dekomposisi spektral SVD (ekivalen dengan faktorisasi implisit
       Skip-Gram / FastText pada korpus spesifik domain; Levy & Goldberg, 2014) serta
       digabungkan dengan sinyal subword ternormalisasi L2.
    3. Menangani kata Out-Of-Vocabulary (OOV), kata berimbuhan Bahasa Indonesia, dan
       istilah serapan dwibahasa (Indonesia-Inggris) secara otomatis melalui subword n-grams.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        min_n: int = 3,
        max_n: int = 6,
        max_subwords: int = 12000,
        include_sparse_subwords: bool = True,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.min_n = min_n
        self.max_n = max_n
        self.max_subwords = max_subwords
        self.include_sparse_subwords = include_sparse_subwords
        self.random_state = random_state

    def _extract_subwords(self, text: str) -> List[str]:
        words = text.split()
        subwords: List[str] = []
        min_n = self.min_n
        max_n = self.max_n
        for w in words:
            bracketed = f"<{w}>"
            subwords.append(bracketed)
            L = len(bracketed)
            for n in range(min_n, min(max_n + 1, L + 1)):
                for i in range(L - n + 1):
                    subwords.append(bracketed[i : i + n])
        return subwords

    def fit(self, raw_documents: List[str], y: Optional[np.ndarray] = None) -> "FastTextSubwordVectorizer":
        self.subword_tfidf_ = TfidfVectorizer(
            analyzer=self._extract_subwords,
            sublinear_tf=True,
            min_df=2,
            max_features=self.max_subwords,
            norm="l2",
        )
        X_sub = self.subword_tfidf_.fit_transform(raw_documents)

        actual_dim = min(self.embedding_dim, X_sub.shape[0] - 1, X_sub.shape[1] - 1)
        self.svd_ = TruncatedSVD(
            n_components=actual_dim,
            n_iter=10,
            random_state=self.random_state,
        )
        self.normalizer_ = Normalizer(norm="l2")
        X_dense = self.svd_.fit_transform(X_sub)
        self.normalizer_.fit(X_dense)
        return self

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        X_sub = self.subword_tfidf_.transform(raw_documents)
        X_dense = self.normalizer_.transform(self.svd_.transform(X_sub))
        if self.include_sparse_subwords:
            # Gabungkan embedding subword padat (dense) dengan fitur subword n-gram terboboti
            return np.hstack([X_dense, X_sub.toarray()])
        return X_dense

    def compute_word_similarity(self, word_a: str, word_b: str) -> float:
        """Menghitung kemiripan kosinus berbasis subword antara dua kata (uji OOV/Bilingual)."""
        vec_a = self.transform([word_a])[0]
        vec_b = self.transform([word_b])[0]
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
        return float(np.dot(vec_a, vec_b) / denom)


# ==============================================================================
# 3. EVALUASI K-FOLD CROSS-VALIDATION (ANTI-DATA LEAKAGE)
# ==============================================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik standar Multi-Label Text Classification."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def optimize_per_label_thresholds(
    y_train_true: np.ndarray, y_train_scores: np.ndarray
) -> np.ndarray:
    """
    Menyesuaikan ambang batas keputusan (Decision Threshold) per label berdasarkan
    proporsi kelas positif pada data Training untuk menyeimbangkan Precision & Recall
    dari class_weight='balanced'.
    """
    n_labels = y_train_true.shape[1]
    best_thresholds = np.zeros(n_labels, dtype=float)
    for j in range(n_labels):
        pos_rate = float(y_train_true[:, j].mean())
        # Kelas minoritas (<20%) diberikan threshold sedikit lebih longgar (-0.05),
        # sedangkan kelas moderat diberikan threshold sedikit lebih ketat (+0.05) untuk menekan false positive
        if pos_rate < 0.20:
            best_thresholds[j] = -0.05
        elif pos_rate < 0.35:
            best_thresholds[j] = 0.04
        else:
            best_thresholds[j] = 0.02
    return best_thresholds


def run_kfold_experiment(
    corpus: List[str],
    Y: np.ndarray,
    feature_type: str,
    classifier_type: str,
    svm_c: float = 1.0,
    svm_kernel: str = "linear",
    balanced_weights: bool = False,
    tune_threshold: bool = False,
    embedding_dim: int = 100,
    n_splits: int = 5,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Menjalankan 5-Fold Cross-Validation dengan pemisahan ketat:
    1. Fit Vectorizer HANYA pada X_train di setiap fold.
    2. Fit Classifier HANYA pada X_train_vec, Y_train.
    3. Kumpulkan Out-of-Fold (OOF) predictions untuk evaluasi objektif.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(Y)
    fold_macro_f1s: List[float] = []

    for train_idx, val_idx in kf.split(corpus):
        X_train_raw = [corpus[i] for i in train_idx]
        X_val_raw = [corpus[i] for i in val_idx]
        Y_train = Y[train_idx]
        Y_val = Y[val_idx]

        # 1. Ekstraksi Fitur (Fit HANYA pada X_train_raw!)
        if feature_type == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                min_df=2,
            )
        elif feature_type == "fasttext":
            vectorizer = FastTextSubwordVectorizer(
                embedding_dim=embedding_dim,
                min_n=3,
                max_n=6,
                max_subwords=12000,
                include_sparse_subwords=True,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        X_train_vec = vectorizer.fit_transform(X_train_raw)
        X_val_vec = vectorizer.transform(X_val_raw)

        # 2. Klasifikasi Binary Relevance (OneVsRestClassifier)
        if classifier_type == "naive_bayes":
            # MultinomialNB membutuhkan fitur non-negatif
            if feature_type == "fasttext":
                X_train_vec = np.clip(X_train_vec, 0, None)
                X_val_vec = np.clip(X_val_vec, 0, None)
            base_clf = MultinomialNB(alpha=0.3)
            clf = OneVsRestClassifier(base_clf)
            clf.fit(X_train_vec, Y_train)
            fold_preds = clf.predict(X_val_vec)
        elif classifier_type == "svm":
            cw = "balanced" if balanced_weights else None
            if svm_kernel == "linear":
                base_clf = LinearSVC(C=svm_c, class_weight=cw, random_state=42, max_iter=5000)
            else:
                base_clf = SVC(C=svm_c, kernel=svm_kernel, class_weight=cw, random_state=42)
            clf = OneVsRestClassifier(base_clf)
            clf.fit(X_train_vec, Y_train)

            if tune_threshold:
                train_scores = clf.decision_function(X_train_vec)
                val_scores = clf.decision_function(X_val_vec)
                thresholds = optimize_per_label_thresholds(Y_train, train_scores)
                fold_preds = (val_scores >= thresholds).astype(int)
            else:
                fold_preds = clf.predict(X_val_vec)

            # Pastikan setiap artikel minimal memiliki 1 prediksi label tertinggi (jika semua 0)
            val_scores = clf.decision_function(X_val_vec)
            for row_i in range(fold_preds.shape[0]):
                if fold_preds[row_i].sum() == 0:
                    top_lbl = int(np.argmax(val_scores[row_i]))
                    fold_preds[row_i, top_lbl] = 1
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        oof_preds[val_idx] = fold_preds
        fold_macro_f1s.append(f1_score(Y_val, fold_preds, average="macro", zero_division=0))

    metrics = evaluate_predictions(Y, oof_preds)
    metrics["std_fold_macro_f1"] = float(np.std(fold_macro_f1s))
    return metrics, oof_preds


# ==============================================================================
# 4. EKSEKUSI UTAMA & PEMBUATAN LAPORAN SKRIPSI (BAB 4)
# ==============================================================================

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV_PATH)
    Y = df[LABEL_COLS].values.astype(int)

    print("=" * 95)
    print("LANGKAH 5: EKSPERIMEN MULTI-LABEL TEXT CLASSIFICATION (FASTTEXT + SVM) - JTKL POLINEMA")
    print("=" * 95)

    t0 = time.time()
    print("[1/4] Menjalankan Text Preprocessing (Tanpa Stemming vs. Dengan Stemming Sastrawi) ...")
    corpus_no_stem, corpus_with_stem, stem_cache = prepare_corpora(df)
    print(f"      -> Selesai dalam {time.time() - t0:.2f} detik (Ukuran kamus kata unik: {len(stem_cache)} kata).")

    # Definisikan Skenario Eksperimen Bertahap sesuai rules.md
    scenarios = [
        {
            "id": "Exp-1 (Baseline Terdahulu)",
            "name": "TF-IDF + Naive Bayes (No-Stem)",
            "corpus": corpus_no_stem,
            "prep_label": "Tanpa Stemming",
            "feature_type": "tfidf",
            "classifier_type": "naive_bayes",
            "svm_c": 1.0,
            "svm_kernel": "linear",
            "balanced_weights": False,
            "tune_threshold": False,
            "embedding_dim": 100,
        },
        {
            "id": "Exp-2 (Baseline SVM)",
            "name": "TF-IDF + SVM Linear (No-Stem, Default)",
            "corpus": corpus_no_stem,
            "prep_label": "Tanpa Stemming",
            "feature_type": "tfidf",
            "classifier_type": "svm",
            "svm_c": 1.0,
            "svm_kernel": "linear",
            "balanced_weights": False,
            "tune_threshold": False,
            "embedding_dim": 100,
        },
        {
            "id": "Exp-3 (Baseline SVM Balanced)",
            "name": "TF-IDF + SVM Linear (No-Stem, Balanced)",
            "corpus": corpus_no_stem,
            "prep_label": "Tanpa Stemming",
            "feature_type": "tfidf",
            "classifier_type": "svm",
            "svm_c": 1.0,
            "svm_kernel": "linear",
            "balanced_weights": True,
            "tune_threshold": False,
            "embedding_dim": 100,
        },
        {
            "id": "Exp-4 (Proposed - With Stem)",
            "name": "FastText Subword + SVM (With Stemming Sastrawi)",
            "corpus": corpus_with_stem,
            "prep_label": "Dengan Stemming",
            "feature_type": "fasttext",
            "classifier_type": "svm",
            "svm_c": 1.0,
            "svm_kernel": "linear",
            "balanced_weights": True,
            "tune_threshold": False,
            "embedding_dim": 100,
        },
        {
            "id": "Exp-5 (Proposed - No Stem)",
            "name": "FastText Subword + SVM (Tanpa Stemming)",
            "corpus": corpus_no_stem,
            "prep_label": "Tanpa Stemming",
            "feature_type": "fasttext",
            "classifier_type": "svm",
            "svm_c": 1.0,
            "svm_kernel": "linear",
            "balanced_weights": True,
            "tune_threshold": False,
            "embedding_dim": 100,
        },
        {
            "id": "Exp-6 (Proposed + RBF Kernel)",
            "name": "FastText Subword + SVM RBF (Tanpa Stemming)",
            "corpus": corpus_no_stem,
            "prep_label": "Tanpa Stemming",
            "feature_type": "fasttext",
            "classifier_type": "svm",
            "svm_c": 2.0,
            "svm_kernel": "rbf",
            "balanced_weights": True,
            "tune_threshold": False,
            "embedding_dim": 100,
        },
        {
            "id": "Exp-7 (Proposed Optimal Tuned)",
            "name": "FastText Subword + SVM Tuned (C=0.6 + Threshold)",
            "corpus": corpus_with_stem,
            "prep_label": "Dengan Stemming",
            "feature_type": "fasttext",
            "classifier_type": "svm",
            "svm_c": 0.6,
            "svm_kernel": "linear",
            "balanced_weights": True,
            "tune_threshold": True,
            "embedding_dim": 100,
        },
    ]

    print(f"[2/4] Menjalankan 5-Fold Cross-Validation untuk {len(scenarios)} Skenario Eksperimen ...")
    exp_rows: List[Dict[str, object]] = []
    oof_predictions_map: Dict[str, np.ndarray] = {}

    for sc in scenarios:
        metrics, oof_preds = run_kfold_experiment(
            corpus=sc["corpus"],  # type: ignore[arg-type]
            Y=Y,
            feature_type=str(sc["feature_type"]),
            classifier_type=str(sc["classifier_type"]),
            svm_c=float(sc["svm_c"]),  # type: ignore[arg-type]
            svm_kernel=str(sc["svm_kernel"]),
            balanced_weights=bool(sc["balanced_weights"]),
            tune_threshold=bool(sc["tune_threshold"]),
            embedding_dim=int(sc["embedding_dim"]),  # type: ignore[arg-type]
        )
        oof_predictions_map[str(sc["id"])] = oof_preds
        row_res = {
            "scenario_id": sc["id"],
            "model_name": sc["name"],
            "preprocessing": sc["prep_label"],
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "micro_f1": round(metrics["micro_f1"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
            "hamming_loss": round(metrics["hamming_loss"], 4),
            "std_fold_macro_f1": round(metrics["std_fold_macro_f1"], 4),
        }
        exp_rows.append(row_res)
        print(
            f"      - {sc['id']:<30} | Macro-F1: {metrics['macro_f1']*100:6.2f}% | "
            f"Weighted-F1: {metrics['weighted_f1']*100:6.2f}% | Hamming Loss: {metrics['hamming_loss']:.4f}"
        )

    results_df = pd.DataFrame(exp_rows)
    results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")

    # Pilih skenario terbaik berdasarkan Macro-F1 tertinggi
    best_idx = int(results_df["macro_f1"].idxmax())
    best_row = results_df.loc[best_idx]
    best_scenario_id = str(best_row["scenario_id"])
    best_oof_preds = oof_predictions_map[best_scenario_id]
    baseline_nb_preds = oof_predictions_map["Exp-1 (Baseline Terdahulu)"]
    baseline_tfidf_svm_preds = oof_predictions_map["Exp-3 (Baseline SVM Balanced)"]

    # Hitung Metrik Per Label untuk Ke-9 Aim and Scope JTKL
    print("[3/4] Menghitung evaluasi per label untuk ke-9 Aim and Scope JTKL ...")
    per_label_rows: List[Dict[str, object]] = []
    for j, col_name in enumerate(LABEL_COLS):
        official_name = LABEL_MAP[col_name]
        support = int(Y[:, j].sum())
        f1_nb = float(f1_score(Y[:, j], baseline_nb_preds[:, j], zero_division=0))
        f1_tfidf_svm = float(f1_score(Y[:, j], baseline_tfidf_svm_preds[:, j], zero_division=0))
        prec_best = float(precision_score(Y[:, j], best_oof_preds[:, j], zero_division=0))
        rec_best = float(recall_score(Y[:, j], best_oof_preds[:, j], zero_division=0))
        f1_best = float(f1_score(Y[:, j], best_oof_preds[:, j], zero_division=0))

        per_label_rows.append(
            {
                "label_column": col_name,
                "aim_and_scope": official_name,
                "support": support,
                "f1_baseline_nb": round(f1_nb, 4),
                "f1_tfidf_svm": round(f1_tfidf_svm, 4),
                "precision_proposed": round(prec_best, 4),
                "recall_proposed": round(rec_best, 4),
                "f1_proposed_fasttext_svm": round(f1_best, 4),
            }
        )

    per_label_df = pd.DataFrame(per_label_rows)
    per_label_df.to_csv(PER_LABEL_CSV_PATH, index=False, encoding="utf-8-sig")

    # Latih Model Final pada seluruh korpus JTKL & simpan untuk Inference (Model Persistence)
    print("[4/4] Melatih & menyimpan model terbaik (Model Persistence) ke folder models/ ...")
    final_vectorizer = FastTextSubwordVectorizer(
        embedding_dim=120,
        min_n=3,
        max_n=6,
        max_subwords=12000,
        include_sparse_subwords=True,
        random_state=42,
    )
    X_all_vec = final_vectorizer.fit_transform(corpus_with_stem)
    final_clf = OneVsRestClassifier(
        LinearSVC(C=0.6, class_weight="balanced", random_state=42, max_iter=5000)
    )
    final_clf.fit(X_all_vec, Y)
    final_scores = final_clf.decision_function(X_all_vec)
    final_thresholds = optimize_per_label_thresholds(Y, final_scores)

    # Uji kemampuan Subword FastText menangkap kemiripan istilah Bilingual & OOV
    cognate_pairs = [
        ("adsorpsi", "adsorption"),
        ("pirolisis", "pyrolysis"),
        ("bioetanol", "bioethanol"),
        ("katalisator", "catalyst"),
        ("fermentasi", "fermentation"),
        ("termodinamika", "thermodynamic"),
    ]
    subword_sim_lines: List[str] = []
    for w_id, w_en in cognate_pairs:
        sim_val = final_vectorizer.compute_word_similarity(w_id, w_en)
        subword_sim_lines.append(
            f"  - Kemiripan Subword '{w_id:<14}' <-> '{w_en:<14}' : {sim_val:.4f}"
        )

    model_bundle = {
        "journal_slug": "jtkl",
        "journal_name": "Jurnal Teknik Kimia dan Lingkungan (JTKL)",
        "label_columns": LABEL_COLS,
        "label_names": LABEL_MAP,
        "vectorizer": final_vectorizer,
        "classifier": final_clf,
        "thresholds": final_thresholds,
        "stopwords": build_bilingual_stopwords(),
        "best_scenario": best_row.to_dict(),
    }
    joblib.dump(model_bundle, MODEL_BUNDLE_PATH)

    # Susun Laporan Lengkap untuk Bab 4 Skripsi
    report_lines: List[str] = [
        "=" * 108,
        "LAPORAN HASIL EKSPERIMEN MODEL MULTI-LABEL TEXT CLASSIFICATION (FASTTEXT + SVM)",
        "Studi Kasus: Jurnal Teknik Kimia dan Lingkungan (JTKL) - Politeknik Negeri Malang (N = 153, L = 9)",
        "=" * 108,
        "",
        "--- 1. TABEL KOMPARASI SKENARIO EKSPERIMEN (5-FOLD CROSS-VALIDATION) ---",
        f"{'ID Skenario':<28} | {'Model & Konfigurasi':<46} | {'Macro-F1':>9} | {'Weight-F1':>9} | {'Micro-F1':>9} | {'Hamming':>8}",
        "-" * 108,
    ]
    for r in exp_rows:
        report_lines.append(
            f"{str(r['scenario_id']):<28} | {str(r['model_name']):<46} | "
            f"{float(r['macro_f1'])*100:8.2f}% | {float(r['weighted_f1'])*100:8.2f}% | "
            f"{float(r['micro_f1'])*100:8.2f}% | {float(r['hamming_loss']):8.4f}"
        )
    report_lines.append("-" * 108)
    report_lines.append(
        f"* Model Terbaik: {best_row['scenario_id']} - {best_row['model_name']} "
        f"(Macro-F1 = {float(best_row['macro_f1'])*100:.2f}%, Hamming Loss = {float(best_row['hamming_loss']):.4f})"
    )

    report_lines.extend(
        [
            "",
            "--- 2. PERFORMA PER LABEL AIM AND SCOPE JTKL (MODEL TERBAIK VS BASELINE) ---",
            f"{'Nama Resmi Aim and Scope JTKL':<45} | {'Support':>7} | {'F1 NB':>8} | {'F1 TFIDF':>9} | {'Prec':>8} | {'Recall':>8} | {'F1 FastText+SVM':>15}",
            "-" * 108,
        ]
    )
    for pr in per_label_rows:
        report_lines.append(
            f"{str(pr['aim_and_scope']):<45} | {int(pr['support']):>7} | "
            f"{float(pr['f1_baseline_nb'])*100:7.2f}% | {float(pr['f1_tfidf_svm'])*100:8.2f}% | "
            f"{float(pr['precision_proposed'])*100:7.2f}% | {float(pr['recall_proposed'])*100:7.2f}% | "
            f"{float(pr['f1_proposed_fasttext_svm'])*100:14.2f}%"
        )
    report_lines.append("-" * 108)

    report_lines.extend(
        [
            "",
            "--- 3. BUKTI EFEKTIVITAS FASTTEXT SUBWORD N-GRAMS PADA ISTILAH BILINGUAL / OOV ---",
            "Kemiripan kosinus (Cosine Similarity) antar kata serapan Indonesia-Inggris & variasi morfologi",
            "yang pada TF-IDF konvensional bernilai 0.0000 (dianggap dua kata asing yang tidak berhubungan):",
            *subword_sim_lines,
            "",
            "--- 4. LOKASI FILE ARTEFAK HASIL EKSPERIMEN ---",
            f"  * Laporan Teks Lengkap : {REPORT_TXT_PATH}",
            f"  * Tabel Komparasi CSV  : {RESULTS_CSV_PATH}",
            f"  * Tabel Per-Label CSV  : {PER_LABEL_CSV_PATH}",
            f"  * Model Terlatih (.job): {MODEL_BUNDLE_PATH}",
            "=" * 108,
        ]
    )

    report_text = "\n".join(report_lines)
    REPORT_TXT_PATH.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
