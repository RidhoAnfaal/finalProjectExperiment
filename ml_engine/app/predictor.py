from pathlib import Path
import re, json, joblib
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class JTKLPredictor :
    def __init__(self, threshold: float = -0.15) :
        root = Path(__file__).resolve().parent.parent
        self.threshold = threshold
        
        # load model bundle berisi vectorizer + svm + daftar label
        bundle = joblib.load(root / 'models' / 'jtkl_model_v1.joblib')
        self.vec, self.clf, self.labels = bundle['vec'], bundle['clf'], bundle['labels']
        
        # load dict cache stemming & stopwords
        self.stem_cache = json.loads((root / 'data' / 'processed' / 'sastrawi_stem_cache.json').read_text(encoding='utf-8'))
        self.stopwords = set(StopWordRemoverFactory().get_stop_words()) | set(ENGLISH_STOP_WORDS)
    
    def _clean_and_stem(self, text: str) -> str:
        tokens = [w for w in re.findall(r'[a-z]{3,}', str(text).lower()) if w not in self.stopwords]
        return ' '.join(self.stem_cache.get(w, w) for w in tokens)

    def predict(self, title: str, abstract: str = "", keywords: str = "") -> dict:
        # Pembobotan domain 
        raw_text = f"{title} {title} {keywords} {keywords} {abstract}"
        clean_text = self._clean_and_stem(raw_text)
        
        # Ekstraksi fitur & kalkulasi margin decision SVM
        x_vec = self.vec.transform([clean_text])
        scores = self.clf.decision_function(x_vec)[0]
        
        # Seleksi label aktif berdasarkan threshold optimal = -0.15
        predicted_topics = [
            self.labels[i] for i, score in enumerate(scores) if score >= self.threshold
        ]
        
        return {
            "topics": predicted_topics if predicted_topics else ["unclassified"],
            "scores": {self.labels[i]: round(float(scores[i]), 4) for i in range(len(self.labels))}
        }
        
if __name__ == '__main__':
    # Inisialisasi model (cukup 1 kali)
    predictor = JTKLPredictor()
    
    # Contoh input artikel baru (dari jurnal / dosen)
    test_title = "Studi Kinetika Reaksi Metanolisis Pembuatan Metil Ester Sulfonat (MES) Menggunakan Reaktor Batch Berpengaduk"
    test_abstract = "Surfactant is a surface active agent which is widely applied in chemical industry related to its ability to stabilize the emulsion between the oil phase and the water phase. Surfactant-based on vegetable oil is a technology developed in the field of surfactant which has been dominated by surfactants made from petroleum. Methyl ester sulfonate (MES) is an anionic surfactant produced by the reaction between fatty acid methyl ester with a sulfonating agent or called as sulfonation reaction. MES undergoes an advanced process called methanolysis and neutralization reactions. This study investigated the effect of reactant molar ratio, reaction time and reaction temperature on MES yield on methanolysis reaction. The highest MES yield of 49.71% was achieved at a reaction temperature of 120°C, the reaction time of 120 minutes and the molar ratio between MES and methanol of 1:3. The rate constants of methanolysis reactions are determined by reacting the reactants in the stirred batch reactor under those operating conditions."
    test_keywords = "reaction kinetics, methanolysis, methyl ester sulfonate, batch reactor, surfactant"
    
    hasil = predictor.predict(title=test_title, abstract=test_abstract, keywords=test_keywords)
   
    print("HASIL KLASIFIKASI MULTI-LABEL OTOMATIS:\n")
    print(f"Judul Artikel   : {test_title[:75]}...")
    print(f"Topik Terpilih  : {hasil['topics']}")
    print("\nDetail Skor Margin SVM per Topik (Threshold >= -0.15):")
    for lbl, score in hasil['scores'].items():
        status = " -> [AKTIF]" if score >= -0.15 else ""
        print(f"{lbl:<26}: {score:>7.4f}{status}")