from pandas.core.reshape import encoding
from matplotlib.pyplot import clf
from pathlib import Path
import re, pandas as pd
import joblib, json
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# import model jtkl 
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | set(ENGLISH_STOP_WORDS)

# normalize authors name
def normalize_name(name: str) -> str:
    words = re.findall(r'[A-Z]+', str(name).upper())
    return words[0] if len(words) == 2 and words[0] == words[1] else ' '.join(words)

# change sinta author to hash map dict
ROOT = Path(__file__).resolve().parent.parent

df_sinta = pd.read_csv(ROOT / 'data' / 'raw' / 'polinema_sinta_authors.csv')
df_art = pd.read_csv(ROOT / 'data' / 'raw' / 'jtkl_articles_archive.csv')
df_sinta['norm_name'] = df_sinta['name'].apply(normalize_name)
sinta_lookup = df_sinta.drop_duplicates('norm_name').set_index('norm_name').to_dict('index')

# load model svm & sastrawi stem
bundle = joblib.load(ROOT / 'models' / 'jtkl_model_v1.joblib')
vec, clf, label_cols = bundle['vec'], bundle['clf'], bundle['labels']
stem_cache = json.loads((ROOT / 'data' / 'processed' / 'sastrawi_stem_cache.json').read_text(encoding='utf-8'))

# function cleaning & stemming text
def clean_and_stem(text: str) -> str:
    tokens = [w for w in re.findall(r'[a-z]{3,}', str(text).lower()) if w not in STOPWORDS]
    return ' '.join(stem_cache.get(w, w) for w in tokens)

# pembobotan teks: Title (2x) + Keywords (2x) + Abstract (1x) konsisten dengan pelatihan model
raw_texts = (df_art['title'].fillna('') + ' ' + df_art['title'].fillna('') + ' ' +
             df_art['keywords'].fillna('') + ' ' + df_art['keywords'].fillna('') + ' ' +
             df_art['abstract'].fillna(''))
clean_texts = raw_texts.apply(clean_and_stem).tolist()

# prediksi multi-label dengan threshold optimal tau = -0.15
preds = (clf.decision_function(vec.transform(clean_texts)) >= -0.15).astype(int)
df_art['predicted_topics'] = [
    ';'.join(label_cols[i] for i, val in enumerate(p) if val == 1) or 'unclassified'
    for p in preds
]

# entity resolution process
# split/pecah nama penulis artikel dari metadata artikel then matching to author profil sinta
import difflib

synced_records = []
for _, row in df_art.iterrows():
    for raw_author in str(row['authors']).split(';'):
        norm = normalize_name(raw_author)
        match = [norm] if norm in sinta_lookup else difflib.get_close_matches(norm, sinta_lookup.keys(), n=1, cutoff=0.92)
        if match: 
            synced_records.append({
                "article_id": row["article_id"], 
                "title": row["title"],
                "year": row["year"], 
                "ojs_author": raw_author.strip(), 
                "sinta_id": sinta_lookup[match[0]]["sinta_id"], 
                "sinta_name": sinta_lookup[match[0]]["name"], 
                "dept": sinta_lookup[match[0]]["dept"],
                "topics": row["predicted_topics"]
            })
            
# store synced output
df_sync = pd.DataFrame(synced_records)
df_sync.to_csv(ROOT / 'data' / 'processed' / 'jtkl_sinta_synced.csv', index=False, encoding='utf-8-sig')
print(f"Total relasi artikel-dosen : {len(df_sync)}\nDosen Polinema unik : {df_sync['sinta_id'].nunique()}\nArtikel dgn dosen Polinema : {df_sync['article_id'].nunique()} dari {len(df_art)} artikel")
print(df_sync[["ojs_author", "sinta_name", "dept"]].head(10).to_string())