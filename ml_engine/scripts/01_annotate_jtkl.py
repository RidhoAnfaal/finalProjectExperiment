r"""
Langkah 4: Dataset Preparation, Multi-Label Annotation & Exploratory Data Analysis (EDA)
Target Dataset : D:\Skripsi_Polinema\ml_engine\data\raw\jtkl_articles_archive.csv
Output Dataset : D:\Skripsi_Polinema\ml_engine\data\processed\jtkl_labeled.csv
Laporan EDA    : D:\Skripsi_Polinema\ml_engine\data\processed\jtkl_eda_summary.json

Taksonomi Ground Truth Mengacu 100% pada 9 Aim and Scope Resmi JTKL Polinema
(https://jurnal.polinema.ac.id/index.php/jtkl/aimandscope):
1. Energy                                       -> label_energy
2. Waste treatment                              -> label_waste_treatment
3. Unit operation                               -> label_unit_operation
4. Thermodynamic                                -> label_thermodynamic
5. Process simulation                           -> label_process_simulation
6. Development and application of new material  -> label_new_material
7. Chemical engineering reaction                -> label_chemical_reaction
8. Biochemical                                  -> label_biochemical
9. Biomass                                      -> label_biomass
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "jtkl_articles_archive.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELED_CSV_PATH = PROCESSED_DIR / "jtkl_labeled.csv"
EDA_JSON_PATH = PROCESSED_DIR / "jtkl_eda_summary.json"
EDA_TXT_PATH = PROCESSED_DIR / "jtkl_eda_report.txt"

# Definisi Resmi 9 Aim and Scope JTKL beserta kamus indikator spesifik (Bilingual ID + EN)
# Catatan Metodologis:
# - Kata generik seperti "material" berdiri sendiri TIDAK dimasukkan agar frasa umum
#   seperti "raw material" atau "materials and methods" tidak memicu false positive.
# - Title & Keywords diberikan bobot utama (Primary Signal), sedangkan Abstract
#   diberikan bobot kontekstual (Secondary Signal).
AIM_AND_SCOPE_TAXONOMY: Dict[str, Dict[str, object]] = {
    "label_energy": {
        "no": 1,
        "official_name": "Energy",
        "primary_patterns": [
            r"\bbiohydrogen\b", r"\bbioethanol\b", r"\bbioetanol\b", r"\bbiodiesel\b",
            r"\bbiogas\b", r"\bbiomethane\b", r"\bsolar cell\b", r"\bdye-sensitized\b",
            r"\bbio-oil\b", r"\bbiooil\b", r"\bsyngas\b", r"\bbriquette\b", r"\bbiobriquette\b",
            r"\bbriket\b", r"\bbiobriket\b", r"\bbiofuel\b", r"\bliquid fuel\b",
            r"\balternative fuel\b", r"\bfuel oil\b", r"\bbiosolar\b", r"\bbutanol\b",
            r"(?<!activation )\benergy\b", r"\benergi(?! aktivasi)\b", r"\bbioenergy\b", r"\bwaste-to-energy\b",
            r"\bcalorific\b", r"\bnilai kalor\b", r"\bdiesel engine\b", r"\bpower plant\b",
            r"\bcoal\b", r"\bbatubara\b", r"\blignite\b", r"\bmetallurgical coke\b",
            r"\brefuse derived fuel\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_waste_treatment": {
        "no": 2,
        "official_name": "Waste treatment",
        "primary_patterns": [
            r"\bwastewater\b", r"\bair limbah\b", r"\blimbah cair\b", r"\beffluent\b",
            r"\bpome\b", r"\bgrey water\b", r"\bleachate\b", r"\blindi\b", r"\bwwtp\b",
            r"\bsludge\b", r"\blumpur aktif\b", r"\bactivated sludge\b", r"\bofmsw\b",
            r"\bwaste treatment\b", r"\bwaste management\b", r"\bpengolahan limbah\b",
            r"\bplastic waste\b", r"\blimbah plastik\b", r"\bmedical mask waste\b",
            r"\bheavy metal\b", r"\blogam berat\b", r"\bchromium\b", r"\bkromium\b",
            r"\bcadmium\b", r"\bcopper\b", r"\bcu\(ii\)\b", r"\bcu removal\b",
            r"\bdye removal\b", r"\bdye waste\b", r"\bremazol\b", r"\bmetil biru\b",
            r"\bmethylene blue\b", r"\bphytoremediation\b", r"\bfitoremediasi\b",
            r"\bbiokonversi sampah\b", r"\bsampah organik\b", r"\borganic waste\b",
            r"\blife cycle assessment\b", r"\blca\b", r"\bco2 emission\b",
            r"\bcapturing co2\b", r"\bammonium removal\b", r"\bnitrate removal\b",
            r"\bwater purification\b", r"\bdrinking water\b", r"\btap water\b",
            r"\boil spill\b", r"\btumpahan minyak\b", r"\bproduced water\b",
            r"\bkompos\b", r"\bcompost\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_unit_operation": {
        "no": 3,
        "official_name": "Unit operation",
        "primary_patterns": [
            r"\badsorption\b", r"\badsorpsi\b", r"\badsorbsi\b", r"\bdesorption\b",
            r"\babsorption\b", r"\babsorpsi\b", r"\babsorbtion\b", r"\bscrubber\b",
            r"\bdistillation\b", r"\bdestilasi\b", r"\bdistilasi\b", r"\bhydrodistillation\b",
            r"\bhydrodistilation\b", r"\bdrying\b", r"\bbio-drying\b", r"\bbiodrying\b",
            r"\bpengeringan\b", r"\bspray drying\b", r"\bvacuum drying\b", r"\btray dryer\b",
            r"\bextraction\b", r"\bekstraksi\b", r"\bsoxhlet\b", r"\bleaching\b",
            r"\bfiltration\b", r"\bultrafiltration\b", r"\bultrafiltrasi\b",
            r"\bpervaporation\b", r"\bmembrane\b", r"\bmembran\b",
            r"\bcoagulation\b", r"\bflocculation\b", r"\bkoagulasi\b", r"\bflokulasi\b",
            r"\belectrocoagulation\b", r"\belektrokoagulasi\b",
            r"\bcrystallization\b", r"\bkristalisasi\b", r"\bprecipitation\b",
            r"\bhydrocyclone\b", r"\bliquid-liquid separation\b", r"\bheat exchanger\b",
            r"\bheat exhanger\b", r"\bfluidized bed\b", r"\bfluidisasi\b",
            r"\bpacked bed\b", r"\bfixed-bed\b", r"\bsettlement tank\b",
            r"\bcolumn filter\b", r"\bagitation\b", r"\baeration\b", r"\bmixing\b",
            r"\bevaporation\b", r"\bevaporasi\b", r"\bpurification\b", r"\bpemurnian\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_thermodynamic": {
        "no": 4,
        "official_name": "Thermodynamic",
        "primary_patterns": [
            r"\bthermodynamic\b", r"\bthermodynamics\b", r"\btermodinamika\b",
            r"\bisotherm\b", r"\bisoterm\b", r"\blangmuir\b", r"\bfreundlich\b",
            r"\bvapor-liquid equilibrium\b", r"\bvapor-liquid equlibria\b",
            r"\bkesetimbangan uap-cair\b", r"\bbinary interaction parameter\b",
            r"\bnrtl\b", r"\buniqlac\b", r"\bunifac\b", r"\bsolubilitas\b",
            r"\bsolubility\b", r"\benthalpy\b", r"\bentalpi\b", r"\bentropy\b",
            r"\bentropi\b", r"\bgibbs\b", r"\bactivation energy\b", r"\benergi aktivasi\b",
            r"\bthermal decomposition\b", r"\bthermal stability\b",
            r"\bthermal energy\b", r"\bheat transfer\b", r"\bperpindahan panas\b",
            r"\bntu\b", r"\bcalorific value\b", r"\bnilai kalor\b", r"\bgcv\b",
        ],
        "abstract_min_hits": 1,
    },
    "label_process_simulation": {
        "no": 5,
        "official_name": "Process simulation",
        "primary_patterns": [
            r"\bsimulation\b", r"\bsimulations\b", r"\bsimulasi\b",
            r"\baspen\b", r"\bhysys\b", r"\bchemcad\b", r"\bcfd\b",
            r"\bcomputational fluid dynamic\b", r"\bcomputional fluid dynamics\b",
            r"\bresponse surface methodology\b", r"\brsm\b", r"\bbox-behnken\b",
            r"\bcentral composite design\b", r"\bplackett-burman\b", r"\btaguchi\b",
            r"\bsimplex lattice design\b", r"\bdesign expert\b",
            r"\bpid controller\b", r"\bpid-controlled\b",
            r"\bpemodelan\b", r"\bmodeling\b", r"\bmodelling\b",
            r"\bnumerical and integral\b",
            r"\banalytical hierarchy process\b", r"\btechno-economic\b",
            r"\beconomic evaluation\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_new_material": {
        "no": 6,
        "official_name": "Development and application of new material",
        "primary_patterns": [
            r"\bactivated carbon\b", r"\bactivated-carbon\b", r"\bkarbon aktif\b",
            r"\bactivated charcoal\b", r"\barang aktif\b", r"\bbiochar\b",
            r"\badsorbent\b", r"\badsorben\b", r"\babsorben\b",
            r"\bcomposite\b", r"\bkomposit\b", r"\bnanoparticle\b", r"\bnanopartikel\b",
            r"\bnanofiber\b", r"\bnanoencapsulation\b", r"\bnanoenkapsulasi\b",
            r"\bnano karbon\b", r"\bhydrogel\b", r"\bhidrogel\b", r"\baerogel\b",
            r"\bbiodegradable plastic\b", r"\bbiodegradable plastics\b",
            r"\bdegradable plastic\b", r"\bplastik biodegradable\b",
            r"\bbiodegradable foam\b", r"\bbodegradable foam\b",
            r"\bpoly\(lactic acid\)\b", r"\bpoli asam laktat\b", r"\bpla\b",
            r"\bpolymer\b", r"\bpolimer\b", r"\bmolecularly imprinted polymer\b",
            r"\bpolyethersulfone\b", r"\bpvdf\b", r"\bpolyamide thin film\b",
            r"\bchitosan\b", r"\bkitosan\b", r"\bsilica\b", r"\bsilika\b",
            r"\bcalcium silicate\b", r"\bkalsium silikat\b", r"\bcalcium carbonate\b",
            r"\bpcc\b", r"\btricalcium phosphate\b", r"\bimplan tulang\b",
            r"\bzeolite\b", r"\bzeolit\b", r"\bsodalite\b", r"\bhydrotalcite\b",
            r"\bconcrete brick\b", r"\bdurable cement\b", r"\bsemen\b",
            r"\bcoating agent\b", r"\bcorrosion\b", r"\bkorosi\b", r"\binhibitor\b",
            r"\bsensor\b", r"\bemulsion\b", r"\bencapsulation\b", r"\bsoap\b", r"\bsabun\b",
            r"\bbio-asphalt\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_chemical_reaction": {
        "no": 7,
        "official_name": "Chemical engineering reaction",
        "primary_patterns": [
            r"\bpyrolysis\b", r"\bco-pyrolysis\b", r"\bpyrolisis\b", r"\bpirolisis\b",
            r"\bgasification\b", r"\bgasifikasi\b", r"\bcarbonization\b", r"\bkarbonisasi\b",
            r"\bcatalyst\b", r"\bcatalysts\b", r"\bcatalytic\b", r"\bkatalis\b",
            r"\bphotocatalysis\b", r"\bfotokatalisis\b", r"\bphotodegradation\b",
            r"\bfotodegradasi\b", r"\btio2\b", r"\bcas\b", r"\bceria zirconia\b",
            r"\besterification\b", r"\besterfication\b", r"\besterifikasi\b",
            r"\btransesterification\b", r"\btransesterifikasi\b", r"\bmethanolysis\b",
            r"\bmetanolisis\b", r"\bacetylation\b", r"\bacetylated\b",
            r"\bsulfonation\b", r"\bsulfonated\b", r"\bsaponification\b",
            r"\bdecarboxylation\b", r"\bdekarboksilasi\b", r"\bcarbonation\b",
            r"\bdelignification\b", r"\bdelignifikasi\b",
            r"\bhydrolysis\b", r"\bhidrolisis\b", r"\borganosolv\b",
            r"\breaction kinetic\w*\b", r"\bkinetic\w* reaction\b", r"\bkinetika reaksi\b", r"\breaction rate\b",
            r"\bchemical activation\b", r"\baktivasi kimia\b", r"\bgrafting\b",
            r"\belectrolysis\b", r"\bplasma\b", r"\bcombustion\b", r"\bpembakaran\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_biochemical": {
        "no": 8,
        "official_name": "Biochemical",
        "primary_patterns": [
            r"\bfermentation\b", r"\bfermentasi\b", r"\banaerobic digestion\b",
            r"\bco-digestion\b", r"\bsaccharomyces\b", r"\baspergillus\b",
            r"\bphanerochaete\b", r"\blactobacillus\b", r"\bbacillus\b",
            r"\bmicrobial\b", r"\bbacteria\b", r"\bbakteri\b", r"\byeast\b",
            r"\benzyme\b", r"\benzim\b", r"\benzymatic\b", r"\bxylanase\b",
            r"\bcellulase\b", r"\bselulase\b", r"\blipase\b", r"\bbiocatalyst\b",
            r"\bimmobilized cell\b", r"\bssf\b", r"\bshf\b", r"\bbioreactor\b",
            r"\bsequencing batch reactor\b", r"\bbiofilm reactor\b",
            r"\bbiological aerated filter\b", r"\bbioactivator\b", r"\bbioaktivator\b",
            r"\bessential oil\b", r"\bminyak atsiri\b", r"\bcitronella oil\b",
            r"\bclove oil\b", r"\bspearmint\b", r"\bminyak kapulaga\b",
            r"\bantimicrobial\b", r"\bantibacterial\b", r"\bantioxidant\b",
            r"\banthocyanin\b", r"\bflavonoid\b", r"\btannin\b", r"\btanin\b",
            r"\bbiocoagulant\b", r"\bbiopestisida\b", r"\bfungal protein\b",
            r"\bblack soldier fly\b",
        ],
        "abstract_min_hits": 2,
    },
    "label_biomass": {
        "no": 9,
        "official_name": "Biomass",
        "primary_patterns": [
            r"\bbiomass\b", r"\bbiomassa\b", r"\blignocellulos\w*\b", r"\blignosellulos\w*\b",
            r"\blignoselulosa\b", r"\bcellulose\b", r"\bselulosa\b", r"\blignin\b",
            r"\bbagasse\b", r"\bbaggase\b", r"\bampas tebu\b",
            r"\bcassava\b", r"\bsingkong\b", r"\btapioca\b", r"\btapioka\b",
            r"\bcoffee husk\b", r"\bspent coffee\b", r"\bkulit kopi\b",
            r"\bsawdust\b", r"\bserbuk gergaji\b", r"\bserbuk kayu\b", r"\bkayu mahoni\b",
            r"\btobacco\b", r"\btembakau\b", r"\bwater hyacinth\b", r"\benceng gondok\b",
            r"\bkelakai\b", r"\brice straw\b", r"\bjerami\b",
            r"\boil palm\b", r"\bopefb\b", r"\boil palm empty fruit\b",
            r"\bkelapa sawit\b", r"\bpalm kernel\b", r"\bcoconut\b", r"\bsabut kelapa\b",
            r"\btempurung kelapa\b", r"\bjengkol\b", r"\btamarind\b", r"\bmatoa\b",
            r"\bsunflower\b", r"\bnavicula\b", r"\bchlorella\b", r"\bmicroalgae\b",
            r"\bbasil\b", r"\bred dragon fruit\b", r"\bbuah naga\b",
            r"\bpandanus\b", r"\bsorghum\b", r"\bporang\b", r"\bmoringa\b",
            r"\bnipah\b", r"\bbeluntas\b", r"\bnyamplung\b", r"\bbanana peel\b",
            r"\bkulit pisang\b", r"\bbiji nangka\b", r"\bjackfruit\b",
            r"\bcandlenut\b", r"\bcangkang kemiri\b", r"\bavocado\b", r"\balpukat\b",
            r"\bseaweed\b", r"\brumput laut\b", r"\bdurian\b", r"\bsago starch\b",
            r"\bpati sagu\b", r"\bnata de coco\b", r"\bpapaya seed\b", r"\bbiji pepaya\b",
            r"\bcorn stover\b", r"\baloe vera\b", r"\bumbi talas\b", r"\btalas\b",
            r"\bakasia\b", r"\bacacia\b", r"\bmanure\b", r"\bcow dung\b",
        ],
        "abstract_min_hits": 2,
    },
}

LABEL_COLUMNS: List[str] = list(AIM_AND_SCOPE_TAXONOMY.keys())


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    """Menghitung berapa banyak pola indikator unik yang cocok pada teks."""
    if not text:
        return 0
    hits = 0
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits += 1
    return hits


def annotate_article_row(row: pd.Series) -> Tuple[Dict[str, int], List[str]]:
    """
    Melakukan pelabelan Multi-Hot (0/1) untuk 1 artikel berdasarkan:
    - Sinyal Primer (Judul + Keywords): 1 kecocokan spesifik langsung mengaktifkan label.
    - Sinyal Sekunder (Abstrak): membutuhkan >= abstract_min_hits indikator unik
      untuk mencegah false-positive akibat kata singgungan umum di abstrak.
    """
    title_kw = f"{row.get('title', '')} ; {row.get('keywords', '')}".lower()
    abstract_txt = str(row.get("abstract", "")).lower()

    labels: Dict[str, int] = {}
    active_names: List[str] = []

    for col_name, cfg in AIM_AND_SCOPE_TAXONOMY.items():
        patterns: List[str] = cfg["primary_patterns"]  # type: ignore[assignment]
        min_abs_hits: int = cfg["abstract_min_hits"]  # type: ignore[assignment]

        primary_hits = count_pattern_hits(title_kw, patterns)
        abstract_hits = count_pattern_hits(abstract_txt, patterns)

        if primary_hits >= 1 or abstract_hits >= min_abs_hits:
            labels[col_name] = 1
            active_names.append(str(cfg["official_name"]))
        else:
            labels[col_name] = 0

    return labels, active_names


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV_PATH)
    print("=" * 85)
    print("LANGKAH 4: MULTI-LABEL GROUND TRUTH ANNOTATION & EDA (9 AIM AND SCOPE JTKL)")
    print("=" * 85)
    print(f"Total Artikel Input : {len(df)} artikel dari {RAW_CSV_PATH.name}")

    label_rows: List[Dict[str, int]] = []
    human_readable_labels: List[str] = []
    label_counts_per_row: List[int] = []

    for _, row in df.iterrows():
        lbl_dict, active_list = annotate_article_row(row)
        label_rows.append(lbl_dict)
        human_readable_labels.append("; ".join(active_list))
        label_counts_per_row.append(sum(lbl_dict.values()))

    label_df = pd.DataFrame(label_rows)
    df_out = pd.concat([df, label_df], axis=1)
    df_out["label_count"] = label_counts_per_row
    df_out["aim_and_scope_labels"] = human_readable_labels

    # Cek apakah ada artikel dengan 0 label
    zero_label_mask = df_out["label_count"] == 0
    zero_count = int(zero_label_mask.sum())
    print(f"Artikel tanpa label (0 label) : {zero_count} artikel")
    if zero_count > 0:
        print("\nDaftar artikel dengan 0 label:")
        for idx, r in df_out[zero_label_mask].iterrows():
            print(f"  - Row {idx} (ID {r['article_id']}): {r['title']} || KW: {r['keywords']}")

    # Hitung metrik dataset Multi-Label standar literatur (Tsoumakas & Katakis)
    n_samples = len(df_out)
    n_labels = len(LABEL_COLUMNS)
    label_cardinality = float(df_out["label_count"].mean())
    label_density = float(label_cardinality / n_labels)

    # Distribusi per label & penyusunan laporan teks
    report_lines: List[str] = [
        "=" * 100,
        "LAPORAN HASIL ANOTASI MULTI-LABEL & EDA - JURNAL TEKNIK KIMIA DAN LINGKUNGAN (JTKL)",
        "=" * 100,
        f"Total Artikel Input           : {n_samples} artikel ({RAW_CSV_PATH.name})",
        f"Artikel Tanpa Label (0 label) : {zero_count} artikel",
        "",
        "--- 1. DISTRIBUSI KE-9 LABEL RESMI AIM AND SCOPE JTKL ---",
        f"{'No':>2} | {'Kolom Label':<26} | {'Nama Aim & Scope Resmi':<44} | {'Jumlah':>6} | {'Persentase':>10}",
        "-" * 100,
    ]

    per_label_stats = {}
    for col_name, cfg in AIM_AND_SCOPE_TAXONOMY.items():
        cnt = int(df_out[col_name].sum())
        pct = cnt / n_samples * 100.0
        no = cfg["no"]
        off_name = str(cfg["official_name"])
        per_label_stats[col_name] = {
            "no": no,
            "official_name": off_name,
            "count": cnt,
            "percentage": round(pct, 2),
        }
        report_lines.append(
            f"{no:>2} | {col_name:<26} | {off_name:<44} | {cnt:>6} | {pct:>9.1f}%"
        )
    report_lines.append("-" * 100)

    # Distribusi jumlah label per artikel (Multi-Label Multiplicity)
    multiplicity_dist = df_out["label_count"].value_counts().sort_index().to_dict()
    report_lines.append("")
    report_lines.append("--- 2. DISTRIBUSI JUMLAH LABEL PER ARTIKEL (MULTIPLICITY) ---")
    for k, v in multiplicity_dist.items():
        report_lines.append(
            f"  - {k} Label sekaligus : {v:3d} artikel ({v / n_samples * 100:5.1f}%)"
        )

    report_lines.extend(
        [
            "",
            "--- 3. METRIK KOMPLEKSITAS DATASET MULTI-LABEL ---",
            f"  * Jumlah Dokumen (N)       : {n_samples}",
            f"  * Jumlah Kelas Label (L)   : {n_labels}",
            f"  * Label Cardinality (LC)   : {label_cardinality:.3f} (rata-rata topik per artikel)",
            f"  * Label Density (LD)       : {label_density:.3f} ({label_density * 100:.1f}%)",
            f"  * Unique Labelsets         : {df_out['aim_and_scope_labels'].nunique()} kombinasi unik",
            "",
            "--- 4. CONTOH 10 ARTIKEL PERTAMA BESERTA HASIL PELABELAN MULTI-HOT ---",
        ]
    )

    for i, r in df_out.head(10).iterrows():
        report_lines.append(f"[{i + 1:02d}] (ID: {r['article_id']}) {r['title']}")
        report_lines.append(f"     Keywords : {r['keywords']}")
        report_lines.append(f"     Labels ({r['label_count']}): {r['aim_and_scope_labels']}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Simpan dataset berlabel, ringkasan JSON, dan laporan TXT
    df_out.to_csv(LABELED_CSV_PATH, index=False, encoding="utf-8-sig")
    eda_summary = {
        "n_samples": n_samples,
        "n_labels": n_labels,
        "zero_label_articles": zero_count,
        "label_cardinality": round(label_cardinality, 4),
        "label_density": round(label_density, 4),
        "unique_labelsets": int(df_out["aim_and_scope_labels"].nunique()),
        "multiplicity_distribution": {str(k): int(v) for k, v in multiplicity_dist.items()},
        "per_label_statistics": per_label_stats,
    }
    EDA_JSON_PATH.write_text(json.dumps(eda_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    EDA_TXT_PATH.write_text(report_text, encoding="utf-8")

    print(f"[OK] Dataset Ground Truth tersimpan di : {LABELED_CSV_PATH}")
    print(f"[OK] Ringkasan statistik JSON di      : {EDA_JSON_PATH}")
    print(f"[OK] Laporan Teks EDA lengkap di      : {EDA_TXT_PATH}")


if __name__ == "__main__":
    main()
