r"""
Skrip Eksperimen Harvesting Artikel per Jurnal - Direktori Jurnal Elektronik Polinema
Target Portal : https://jurnal.polinema.ac.id/
Lokasi File   : D:\Skripsi_Polinema\ml_engine\scripts\harvest_journal.py

Fitur Utama:
1. Mendukung 22+ Jurnal Polinema berdasarkan slug resmi OJS.
2. Menarik SELURUH Volume/Issue (lintas halaman arsip) dan SELURUH Artikel di dalamnya.
3. Mengekstrak metadata lengkap untuk kebutuhan Multi-Label Text Classification (FastText + SVM):
   - Judul (Title), Abstrak (Abstract), Kata Kunci (Keywords)
   - Penulis (Authors) & Afiliasi Institusi (untuk Entity Resolution Dosen Polinema)
   - Volume, Nomor (Issue), Tahun (Year), Tanggal Terbit, DOI, URL Artikel, & Link PDF.
4. Menyediakan 2 metode akuisisi data:
   - Mode 'archive' (Default): Deep HTML Scraping bertingkat (Archive -> Semua Volume -> Semua Artikel).
   - Mode 'oai'              : Protokol resmi OAI-PMH (oai_dc + resumptionToken).

Cara Pakai :
  .\venv\Scripts\python.exe scripts\harvest_journal.py --journal jtkl --mode archive
  .\venv\Scripts\python.exe scripts\harvest_journal.py --journal jtkl --mode oai
  .\venv\Scripts\python.exe scripts\harvest_journal.py --list
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Matikan peringatan SSL lokal agar output terminal bersih
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://jurnal.polinema.ac.id/index.php"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 2. Daftar Katalog Resmi 22 Jurnal di Direktori Jurnal Polinema (+1 Jurnal Tambahan di Portal)
POLINEMA_JOURNALS: Dict[str, Dict[str, str]] = {
    "jtkl": {
        "no": "1",
        "slug": "jtkl",
        "name": "Jurnal Teknik Kimia dan Lingkungan (JTKL)",
    },
    "jraam": {
        "no": "2",
        "slug": "JRAAM",
        "name": "Jurnal Riset dan Aplikasi: Akuntansi dan Manajemen (JRAAM)",
    },
    "elposys": {
        "no": "3",
        "slug": "elposys",
        "name": "Elposys: Jurnal Sistem Kelistrikan",
    },
    "elkolind": {
        "no": "4",
        "slug": "elkolind",
        "name": "Jurnal Elektronika dan Otomasi Industri (ELKOLIND)",
    },
    "jip": {
        "no": "5",
        "slug": "jip",
        "name": "JIP (Jurnal Informatika Polinema)",
    },
    "jetm": {
        "no": "6",
        "slug": "JETM",
        "name": "Jurnal Energi dan Teknologi Manufaktur (JETM)",
    },
    "jartel": {
        "no": "7",
        "slug": "jartel",
        "name": "Jurnal Jaringan Telekomunikasi (JARTEL)",
    },
    "distilat": {
        "no": "8",
        "slug": "distilat",
        "name": "DISTILAT: Jurnal Teknologi Separasi",
    },
    "eltek": {
        "no": "9",
        "slug": "eltek",
        "name": "Jurnal ELTEK",
    },
    "prokons": {
        "no": "10",
        "slug": "PROKONS",
        "name": "Prokons: Jurnal Teknik Sipil",
    },
    "ijfte": {
        "no": "11",
        "slug": "IJFTE",
        "name": "International Journal of Frontier Technology and Engineering (IJFTE)",
    },
    "jtia": {
        "no": "12",
        "slug": "jtia",
        "name": "Jurnal Teknik Ilmu dan Aplikasi (JTIA)",
    },
    "jabh": {
        "no": "13",
        "slug": "jabh",
        "name": "Jurnal Akuntansi Bisnis dan Humaniora (JABH)",
    },
    "jab": {
        "no": "14",
        "slug": "jab",
        "name": "Jurnal Aplikasi Bisnis (JAB)",
    },
    "jpkm": {
        "no": "15",
        "slug": "jpkm",
        "name": "Jurnal Pengabdian Polinema Kepada Masyarakat (JPPKM)",
    },
    "abdimas": {
        "no": "16",
        "slug": "abdimas",
        "name": "Jurnal Pengabdian Kepada Masyarakat (Abdimas)",
    },
    "adbis": {
        "no": "17",
        "slug": "adbis",
        "name": "Jurnal Administrasi dan Bisnis (ADBIS)",
    },
    "jlt": {
        "no": "18",
        "slug": "jlt",
        "name": "Jurnal Linguistik Terapan (JLT)",
    },
    "jtim": {
        "no": "19",
        "slug": "jtim",
        "name": "Jurnal Informatika dan Multimedia (JTIM)",
    },
    "j-indeks": {
        "no": "20",
        "slug": "j-indeks",
        "name": "Jurnal Pengabdian pada Masyarakat Ilmu Pengetahuan dan Teknologi Terintegrasi (J-INDEKS)",
    },
    "j-meeg": {
        "no": "21",
        "slug": "j-meeg",
        "name": "Jurnal Teknik Mesin (J-MEEG)",
    },
    "jaeb": {
        "no": "22",
        "slug": "jaeb",
        "name": "Jurnal Akuntansi dan Ekonomi Bisnis (JAEB)",
    },
    "jos-mrk": {
        "no": "23",
        "slug": "jos-mrk",
        "name": "Jurnal Online Skripsi Manajemen Rekayasa Konstruksi (JOS-MRK)",
    },
}


def create_http_session() -> requests.Session:
    """Membuat HTTP session dengan retry otomatis dan User-Agent browser."""
    session = requests.Session()
    retry_strategy = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def clean_whitespace(text: Optional[str]) -> str:
    """Merapikan spasi berlebih, tab, dan baris baru pada teks."""
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def extract_year(date_str: str) -> str:
    """Mengekstrak tahun 4 digit (YYYY) dari string tanggal."""
    if not date_str:
        return ""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", str(date_str))
    return match.group(1) if match else ""


# ==============================================================================
# METODE 1: DEEP ARCHIVE HTML SCRAPER (SEMUA VOLUME/ISSUE -> SEMUA ARTIKEL)
# ==============================================================================

def get_all_issue_urls(session: requests.Session, slug: str) -> List[Tuple[str, str]]:
    """
    Menelusuri halaman /issue/archive (termasuk paginasi halaman 2, 3, dst.)
    untuk mengumpulkan seluruh link Volume/Issue beserta judul edisinya.
    """
    visited_archive_pages = set()
    issues_found: Dict[str, str] = {}  # issue_url -> issue_label
    page_num = 1

    while True:
        if page_num == 1:
            archive_url = f"{BASE_URL}/{slug}/issue/archive"
        else:
            archive_url = f"{BASE_URL}/{slug}/issue/archive/{page_num}"

        if archive_url in visited_archive_pages:
            break
        visited_archive_pages.add(archive_url)

        try:
            resp = session.get(archive_url, verify=False, timeout=25)
            if resp.status_code != 200:
                break
        except Exception as exc:
            print(f"  [!] Gagal membuka halaman arsip {archive_url}: {exc}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        new_issues_on_page = 0

        # Cari semua blok atau link yang mengarah ke /issue/view/{id}
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if f"/{slug}/issue/view/" in href or "/issue/view/" in href:
                # Pastikan bukan link galley issue
                tail = href.split("/issue/view/")[-1].strip("/")
                if "/" in tail:
                    continue
                label = clean_whitespace(a_tag.get_text())
                # Ambil label yang informatif (Vol / No / Tahun)
                parent_summary = a_tag.find_parent("div", class_="obj_issue_summary")
                if parent_summary:
                    series_div = parent_summary.find(class_="series")
                    if series_div:
                        series_txt = clean_whitespace(series_div.get_text())
                        if series_txt and series_txt not in label:
                            label = f"{label} ({series_txt})"

                if href not in issues_found:
                    issues_found[href] = label or f"Issue {tail}"
                    new_issues_on_page += 1
                elif len(label) > len(issues_found[href]):
                    issues_found[href] = label

        # Cek apakah ada link menuju halaman arsip berikutnya (/issue/archive/{page_num + 1})
        has_next_page = any(
            f"/issue/archive/{page_num + 1}" in a["href"]
            for a in soup.find_all("a", href=True)
        )
        if not has_next_page or new_issues_on_page == 0:
            break
        page_num += 1

    return list(issues_found.items())


def get_articles_from_issue(
    session: requests.Session, issue_url: str, issue_label: str
) -> List[Dict[str, str]]:
    """
    Membuka satu halaman Volume/Issue (/issue/view/{id}) dan mengambil semua
    daftar artikel di dalam volume tersebut beserta fallback metadata dasarnya.
    """
    articles_in_issue: Dict[str, Dict[str, str]] = {}
    try:
        resp = session.get(issue_url, verify=False, timeout=25)
        if resp.status_code != 200:
            return []
    except Exception as exc:
        print(f"  [!] Gagal membaca volume {issue_url}: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Ambil judul volume dari halaman issue jika ada
    page_issue_title = soup.find("h1")
    resolved_issue_label = (
        clean_whitespace(page_issue_title.get_text()) if page_issue_title else issue_label
    )

    # Strategi 1: Struktur standar OJS 3 (div.obj_article_summary)
    summary_blocks = soup.find_all("div", class_="obj_article_summary")
    for block in summary_blocks:
        title_node = (
            block.find(["h3", "h4", "div"], class_="title")
            or block.find("a")
        )
        if not title_node:
            continue
        link_node = title_node if title_node.name == "a" else title_node.find("a", href=True)
        if not link_node or "/article/view/" not in link_node.get("href", ""):
            continue

        art_url = link_node["href"].strip()
        article_id = art_url.split("/article/view/")[-1].strip("/").split("/")[0]

        title_text = clean_whitespace(link_node.get_text())
        author_node = block.find("div", class_="authors")
        authors_text = clean_whitespace(author_node.get_text()) if author_node else ""

        pdf_node = block.find("a", class_=re.compile(r"obj_galley_link"))
        pdf_url = pdf_node["href"].strip() if (pdf_node and pdf_node.has_attr("href")) else ""

        articles_in_issue[art_url] = {
            "article_id": article_id,
            "article_url": art_url,
            "issue_url": issue_url,
            "issue_label": resolved_issue_label,
            "fallback_title": title_text,
            "fallback_authors": authors_text,
            "fallback_pdf_url": pdf_url,
        }

    # Strategi 2: Fallback untuk tema OJS 2 / variasi template lain
    if not articles_in_issue:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if "/article/view/" in href:
                tail_parts = href.split("/article/view/")[-1].strip("/").split("/")
                # Hanya ambil link utama artikel (/article/view/ID), bukan galley (/article/view/ID/GALLEY)
                if len(tail_parts) == 1 and tail_parts[0].isdigit():
                    art_id = tail_parts[0]
                    txt = clean_whitespace(a_tag.get_text())
                    if href not in articles_in_issue and txt and txt.upper() not in ("PDF", "HTML", "ABSTRACT"):
                        articles_in_issue[href] = {
                            "article_id": art_id,
                            "article_url": href,
                            "issue_url": issue_url,
                            "issue_label": resolved_issue_label,
                            "fallback_title": txt,
                            "fallback_authors": "",
                            "fallback_pdf_url": "",
                        }

    return list(articles_in_issue.values())


def parse_article_detail(
    session: requests.Session,
    article_stub: Dict[str, str],
    journal_slug: str,
    journal_name: str,
) -> Dict[str, str]:
    """
    Membuka halaman detail artikel (/article/view/{id}) untuk mengekstrak:
    Judul, Abstrak lengkap, Keywords, Penulis, Afiliasi, Tahun, Volume, Issue, DOI, dan PDF URL.
    """
    art_url = article_stub["article_url"]
    record = {
        "journal_slug": journal_slug,
        "journal_name": journal_name,
        "article_id": article_stub.get("article_id", ""),
        "volume": "",
        "issue": "",
        "issue_label": article_stub.get("issue_label", ""),
        "year": "",
        "pub_date": "",
        "title": article_stub.get("fallback_title", ""),
        "abstract": "",
        "keywords": "",
        "authors": article_stub.get("fallback_authors", ""),
        "affiliations": "",
        "language": "",
        "doi": "",
        "pages": "",
        "article_url": art_url,
        "pdf_url": article_stub.get("fallback_pdf_url", ""),
    }

    try:
        resp = session.get(art_url, verify=False, timeout=25)
        if resp.status_code != 200:
            return record
    except Exception:
        return record

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. Ekstraksi dari Meta Tags standar OJS (Dublin Core DC.* & Google Scholar citation_*)
    def get_meta_first(*names: str) -> str:
        for name in names:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                val = clean_whitespace(tag["content"])
                if val:
                    return val
        return ""

    def get_meta_all(*names: str) -> List[str]:
        values: List[str] = []
        for name in names:
            for tag in soup.find_all("meta", attrs={"name": name}):
                val = clean_whitespace(tag.get("content", ""))
                if val and val not in values:
                    values.append(val)
        return values

    meta_title = get_meta_first("citation_title", "DC.Title")
    if meta_title:
        record["title"] = meta_title

    # Abstrak dari DC.Description atau HTML body <section class="item abstract">
    meta_abstract = get_meta_first("DC.Description", "description")
    html_abstract = ""
    abstract_section = (
        soup.find("section", class_=re.compile(r"\babstract\b"))
        or soup.find("div", id="articleAbstract")
        or soup.find("div", class_=re.compile(r"\babstract\b"))
    )
    if abstract_section:
        # Buang heading "Abstract" / "Abstrak" di dalam section
        for heading in abstract_section.find_all(["h2", "h3", "h4", "strong"], class_=re.compile(r"label|title")):
            heading.decompose()
        html_abstract = clean_whitespace(abstract_section.get_text())
        html_abstract = re.sub(r"^(Abstract|Abstrak|ABSTRACT|ABSTRAK)\s*[:\-]?\s*", "", html_abstract)

    # Pilih abstrak yang paling lengkap (kadang meta tag terpotong atau sebaliknya)
    record["abstract"] = html_abstract if len(html_abstract) > len(meta_abstract) else meta_abstract

    # Keywords dari meta tag + HTML body
    kw_list = get_meta_all("citation_keywords", "DC.Subject")
    if not kw_list:
        kw_section = soup.find("section", class_=re.compile(r"\bkeywords\b"))
        if kw_section:
            val_span = kw_section.find(class_="value") or kw_section
            raw_kw = clean_whitespace(val_span.get_text())
            raw_kw = re.sub(r"^(Keywords|Kata Kunci)\s*[:\-]?\s*", "", raw_kw, flags=re.I)
            kw_list = [clean_whitespace(k) for k in re.split(r"[;,]", raw_kw) if clean_whitespace(k)]
    # Bersihkan jika satu tag berisi beberapa keyword dipisah koma/titik koma
    normalized_kw: List[str] = []
    for kw in kw_list:
        for part in re.split(r"[;]", kw):
            p = clean_whitespace(part)
            if p and p not in normalized_kw:
                normalized_kw.append(p)
    record["keywords"] = "; ".join(normalized_kw)

    # Penulis & Afiliasi
    authors_meta = get_meta_all("citation_author", "DC.Creator.PersonalName")
    if authors_meta:
        record["authors"] = "; ".join(authors_meta)

    affil_meta = get_meta_all("citation_author_institution")
    if not affil_meta:
        affil_spans = soup.find_all("span", class_="affiliation")
        affil_meta = [
            clean_whitespace(s.get_text())
            for s in affil_spans
            if clean_whitespace(s.get_text())
        ]
    # Hilangkan duplikat dengan tetap menjaga urutan
    unique_affils = list(dict.fromkeys(affil_meta))
    record["affiliations"] = "; ".join(unique_affils)

    # Volume, Issue, Tanggal, Tahun
    record["volume"] = get_meta_first("citation_volume", "DC.Source.Volume")
    record["issue"] = get_meta_first("citation_issue", "DC.Source.Issue")
    pub_date = get_meta_first("DC.Date.issued", "citation_date", "DC.Date.created")
    record["pub_date"] = pub_date.replace("/", "-") if pub_date else ""
    # Prioritaskan tahun dari nama edisi/volume resmi (misal "Vol. 4 No. 2 (2020)")
    # agar tidak terdistorsi jika admin OJS melakukan re-publish di tahun berikutnya.
    record["year"] = (
        extract_year(record["issue_label"])
        or extract_year(record["pub_date"])
    )

    # Bahasa, DOI, Halaman, PDF URL
    record["language"] = get_meta_first("DC.Language", "citation_language")
    record["doi"] = get_meta_first("citation_doi", "DC.Identifier.DOI")

    first_page = get_meta_first("citation_firstpage")
    last_page = get_meta_first("citation_lastpage")
    if first_page and last_page:
        record["pages"] = f"{first_page}-{last_page}"
    else:
        record["pages"] = get_meta_first("DC.Identifier.pageNumber")

    meta_pdf = get_meta_first("citation_pdf_url")
    if meta_pdf:
        record["pdf_url"] = meta_pdf

    return record


def harvest_via_archive(
    journal_slug: str, journal_name: str, max_workers: int = 6, max_volumes: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Menjalankan Deep HTML Harvesting untuk 1 jurnal lintas seluruh Volume/Issue.
    """
    session = create_http_session()
    print(f"[1/3] Menelusuri daftar Volume/Issue di: {BASE_URL}/{journal_slug}/issue/archive ...")
    issues = get_all_issue_urls(session, journal_slug)
    if max_volumes and max_volumes > 0:
        issues = issues[:max_volumes]

    print(f"      -> Ditemukan {len(issues)} Volume/Issue.")
    if not issues:
        return []

    print(f"[2/3] Mengumpulkan daftar artikel dari {len(issues)} Volume/Issue ...")
    all_article_stubs: List[Dict[str, str]] = []
    seen_article_urls = set()

    for idx, (issue_url, issue_label) in enumerate(issues, start=1):
        stubs = get_articles_from_issue(session, issue_url, issue_label)
        added = 0
        for stub in stubs:
            if stub["article_url"] not in seen_article_urls:
                seen_article_urls.add(stub["article_url"])
                all_article_stubs.append(stub)
                added += 1
        print(f"      - ({idx:02d}/{len(issues):02d}) {issue_label:<45} : {added} artikel")

    total_articles = len(all_article_stubs)
    print(f"\n[3/3] Mengekstrak metadata detail & abstrak dari {total_articles} artikel (workers={max_workers}) ...")

    records: List[Dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                parse_article_detail, session, stub, journal_slug, journal_name
            ): i
            for i, stub in enumerate(all_article_stubs)
        }
        completed = 0
        ordered_results: Dict[int, Dict[str, str]] = {}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            try:
                ordered_results[idx] = future.result()
            except Exception as exc:
                print(f"  [!] Error pada artikel #{idx + 1}: {exc}")
            if completed % 25 == 0 or completed == total_articles:
                print(f"      -> Progress ekstraksi detail: {completed}/{total_articles} artikel selesai.")

    for i in range(total_articles):
        if i in ordered_results:
            records.append(ordered_results[i])

    return records


# ==============================================================================
# METODE 2: OAI-PMH PROTOCOL HARVESTER (oai_dc + resumptionToken)
# ==============================================================================

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def harvest_via_oai(journal_slug: str, journal_name: str) -> List[Dict[str, str]]:
    """
    Menarik seluruh artikel dari satu jurnal menggunakan protokol resmi OAI-PMH
    (mendukung paginasi otomatis via resumptionToken).
    """
    session = create_http_session()
    oai_endpoint = f"{BASE_URL}/{journal_slug}/oai"
    params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
    records: List[Dict[str, str]] = []
    batch_num = 1

    print(f"[OAI-PMH] Menghubungi endpoint: {oai_endpoint} ...")

    while True:
        try:
            resp = session.get(oai_endpoint, params=params, verify=False, timeout=30)
            if resp.status_code != 200:
                print(f"  [!] Server merespons HTTP {resp.status_code}")
                break
        except Exception as exc:
            print(f"  [!] Koneksi OAI-PMH gagal: {exc}")
            break

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            print(f"  [!] Gagal mem-parsing XML OAI-PMH: {exc}")
            break

        list_records = root.find("oai:ListRecords", OAI_NS)
        if list_records is None:
            break

        batch_count = 0
        for rec in list_records.findall("oai:record", OAI_NS):
            header = rec.find("oai:header", OAI_NS)
            if header is not None and header.attrib.get("status") == "deleted":
                continue

            meta = rec.find("oai:metadata/oai_dc:dc", OAI_NS)
            if meta is None:
                continue

            titles = [clean_whitespace(el.text) for el in meta.findall("dc:title", OAI_NS) if el.text]
            creators = [clean_whitespace(el.text) for el in meta.findall("dc:creator", OAI_NS) if el.text]
            subjects = [clean_whitespace(el.text) for el in meta.findall("dc:subject", OAI_NS) if el.text]
            descriptions = [clean_whitespace(el.text) for el in meta.findall("dc:description", OAI_NS) if el.text]
            dates = [clean_whitespace(el.text) for el in meta.findall("dc:date", OAI_NS) if el.text]
            identifiers = [clean_whitespace(el.text) for el in meta.findall("dc:identifier", OAI_NS) if el.text]
            sources = [clean_whitespace(el.text) for el in meta.findall("dc:source", OAI_NS) if el.text]
            languages = [clean_whitespace(el.text) for el in meta.findall("dc:language", OAI_NS) if el.text]

            article_url = next((i for i in identifiers if "/article/view/" in i), "")
            doi = next((i for i in identifiers if i.startswith("10.") or "doi.org/" in i), "")
            article_id = (
                article_url.split("/article/view/")[-1].strip("/").split("/")[0]
                if article_url
                else ""
            )

            # Ekstrak Volume & Issue dari dc:source (misal: "Vol. 10 No. 2 (2024)")
            source_info = next((s for s in sources if "Vol" in s or "No" in s), "")
            vol_match = re.search(r"Vol\.?\s*(\d+)", source_info, re.I)
            iss_match = re.search(r"No\.?\s*(\d+)", source_info, re.I)
            pub_date = dates[0] if dates else ""

            records.append(
                {
                    "journal_slug": journal_slug,
                    "journal_name": journal_name,
                    "article_id": article_id,
                    "volume": vol_match.group(1) if vol_match else "",
                    "issue": iss_match.group(1) if iss_match else "",
                    "issue_label": source_info,
                    "year": extract_year(source_info) or extract_year(pub_date),
                    "pub_date": pub_date,
                    "title": titles[0] if titles else "",
                    "abstract": max(descriptions, key=len) if descriptions else "",
                    "keywords": "; ".join(subjects),
                    "authors": "; ".join(creators),
                    "affiliations": "",
                    "language": languages[0] if languages else "",
                    "doi": doi,
                    "pages": "",
                    "article_url": article_url,
                    "pdf_url": "",
                }
            )
            batch_count += 1

        print(f"  -> Batch #{batch_num}: berhasil menarik {batch_count} artikel aktif (Total: {len(records)})")

        token_el = list_records.find("oai:resumptionToken", OAI_NS)
        if token_el is not None and token_el.text and token_el.text.strip():
            params = {"verb": "ListRecords", "resumptionToken": token_el.text.strip()}
            batch_num += 1
        else:
            break

    return records


# ==============================================================================
# FUNGSI UTAMA & CLI
# ==============================================================================

def resolve_journal(query_key: str) -> Tuple[str, str]:
    """Mencocokkan input user (nomor urut atau kode slug) dengan daftar jurnal Polinema."""
    key_lower = query_key.strip().lower()
    if key_lower in POLINEMA_JOURNALS:
        info = POLINEMA_JOURNALS[key_lower]
        return info["slug"], info["name"]

    for info in POLINEMA_JOURNALS.values():
        if info["no"] == key_lower or info["slug"].lower() == key_lower:
            return info["slug"], info["name"]

    # Jika tidak ada di kamus namun user memasukkan slug kustom
    return query_key.strip(), f"Jurnal Polinema ({query_key.strip()})"


def print_journal_catalog() -> None:
    """Menampilkan tabel 22+ jurnal di Direktori Jurnal Polinema."""
    print("\n=== DAFTAR JURNAL DIREKTORI ELEKTRONIK POLINEMA ===")
    print(f"{'No':>3} | {'Kode Slug':<10} | {'Nama Jurnal'}")
    print("-" * 85)
    for info in POLINEMA_JOURNALS.values():
        print(f"{info['no']:>3} | {info['slug']:<10} | {info['name']}")
    print("-" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvester Artikel Lintas Volume per Jurnal - OJS Polinema"
    )
    parser.add_argument(
        "--journal",
        "-j",
        type=str,
        default="jtkl",
        help="Kode slug jurnal atau nomor urut (1-22). Default: jtkl",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["archive", "oai"],
        default="archive",
        help="Metode harvesting: 'archive' (Deep HTML semua volume) atau 'oai' (OAI-PMH XML). Default: archive",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=6,
        help="Jumlah thread paralel saat mengambil detail artikel pada mode archive (Default: 6)",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Batasi jumlah volume yang diambil (opsional, kosongkan untuk ambil SEMUA volume)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Tampilkan daftar kode 22 jurnal Polinema lalu keluar",
    )

    args = parser.parse_args()

    if args.list:
        print_journal_catalog()
        return

    slug, journal_name = resolve_journal(args.journal)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"HARVESTING DATA ARTIKEL JURNAL POLINEMA")
    print(f"Target Jurnal : {journal_name} (slug: {slug})")
    print(f"Metode        : {args.mode.upper()}")
    print("=" * 80)

    start_time = time.time()

    if args.mode == "archive":
        records = harvest_via_archive(
            journal_slug=slug,
            journal_name=journal_name,
            max_workers=args.workers,
            max_volumes=args.max_volumes,
        )
    else:
        records = harvest_via_oai(journal_slug=slug, journal_name=journal_name)

    elapsed = time.time() - start_time

    if not records:
        print("\n[!] Tidak ada data artikel yang berhasil ditarik.")
        sys.exit(1)

    df = pd.DataFrame(records)
    # Hapus duplikat berdasarkan article_url jika ada
    df = df.drop_duplicates(subset=["article_url"], keep="first").reset_index(drop=True)

    output_filename = f"{slug.lower()}_articles_{args.mode}.csv"
    output_path = RAW_DATA_DIR / output_filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Statistik Kualitas Data hasil Harvesting
    total_rows = len(df)
    non_empty_abstract = int((df["abstract"].fillna("").str.len() > 20).sum())
    non_empty_keywords = int((df["keywords"].fillna("").str.len() > 0).sum())
    unique_volumes = df["issue_label"].nunique()
    years = sorted([y for y in df["year"].unique() if str(y).strip()])

    print("\n" + "=" * 80)
    print("RINGKASAN HASIL HARVESTING (QUALITY CHECK)")
    print("=" * 80)
    print(f"Jurnal                : {journal_name}")
    print(f"Waktu Eksekusi        : {elapsed:.2f} detik")
    print(f"Total Volume/Issue    : {unique_volumes} edisi")
    print(f"Rentang Tahun Terbit  : {years[0] if years else '-'} s/d {years[-1] if years else '-'}")
    print(f"Total Artikel         : {total_rows} artikel")
    print(f"Artikel dgn Abstrak   : {non_empty_abstract}/{total_rows} ({non_empty_abstract / total_rows * 100:.1f}%)")
    print(f"Artikel dgn Keywords  : {non_empty_keywords}/{total_rows} ({non_empty_keywords / total_rows * 100:.1f}%)")
    print(f"Lokasi File Output    : {output_path}")
    print("=" * 80)

    print("\nContoh 3 Artikel Pertama:")
    for i, row in df.head(3).iterrows():
        print(f"[{i + 1}] ({row['year']} - Vol.{row['volume']} No.{row['issue']}) {row['title'][:85]}...")
        print(f"    Penulis  : {row['authors'][:80]}")
        print(f"    Keywords : {row['keywords'][:80]}")
        print(f"    Abstrak  : {row['abstract'][:120]}...\n")


if __name__ == "__main__":
    main()
