"""
Script di sini berfungsi untuk mendapatkan data dosen/researcher Polinema dari SINTA, metadata yang diambil :
- sinta_id: ID unik dosen/researcher di SINTA
- name: Nama dosen/researcher
- dept: Departemen dosen/researcher
- scopus_h_index: Scopus H-Index dosen/researcher
- gs_h_index: Google Scholar H-Index dosen/researcher
- sinta_score_3yr: SINTA Score 3 tahun terakhir
- sinta_score_overall: SINTA Score keseluruhan

"""

from matplotlib.markers import CARETDOWN
from sklearn.utils.extmath import cartesian
from scipy.stats import false_discovery_control
from requests import session
from scipy import stats
from pathlib import Path
import requests, urllib3
from bs4 import BeautifulSoup

# sett endpoint 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
SINTA_URL = "https://sinta.kemdiktisaintek.go.id/affiliations/authors/413"

# test connection 
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36"}
resp = requests.get(SINTA_URL, params={"page": 1}, headers=HEADERS, verify=False, timeout=25)
soup = BeautifulSoup(resp.text, "html.parser")
author_cards = soup.find_all("div", class_="au-item")
print(f"Status HTTP: {resp.status_code} | Jumlah Dosen di Halaman 1: {len(author_cards)}")

# test liat struktur html 
# for tag in author_cards[0].find_all(['div', 'a', 'span'], class_=True):
#     print(f'{tag.name:<5} | class={tag.get('class')} | text={' '.join(tag.get_text().split())[:70]}')

# extracting a card
def parse_author_card(card) -> dict:
    stats = [s.get_text(strip=True) for s in card.find_all('div', class_='stat-num')]
    scopus_h = card.find('span', class_='text-warning')
    gs_h = card.find('span', class_='text-success')
    return {"sinta_id": card.find("div", class_="profile-id").get_text(strip=True).replace("ID :", "").strip(), "name": card.find("div", class_="profile-name").get_text(strip=True), "dept": card.find("div", class_="profile-dept").get_text(strip=True), "scopus_h_index": scopus_h.get_text(strip=True).split(":")[-1].strip() if scopus_h else "0", "gs_h_index": gs_h.get_text(strip=True).split(":")[-1].strip() if gs_h else "0", "sinta_score_3yr": stats[0] if len(stats) > 0 else "0", "sinta_score_overall": stats[1] if len(stats) > 1 else "0"}
# test extraction
# print("\nContoh Hasil Ekstraksi Dosen #1:")
# print(parse_author_card(author_cards[0]))

# get all polinema researchers data
import pandas as pd

all_authors, session = [], requests.Session()
for page in range(1, 79):
    r = session.get(SINTA_URL, params={'page': page}, headers=HEADERS, verify=False, timeout=25)
    cards = BeautifulSoup(r.text, 'html.parser').find_all('div', class_='au-item')
    all_authors.extend([parse_author_card(c) for c in cards]); print(f'Halaman {page:02d}/78 selesai | total : {len(all_authors)} dosen/researcher')

# store data
df_authors = pd.DataFrame(all_authors).drop_duplicates(subset=['sinta_id'])
OUT_CSV = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'polinema_sinta_authors.csv'
df_authors.to_csv(OUT_CSV, index=False, encoding='utf-8-sig'); print(f"\nberhasil menyimpan {len(df_authors)} profil dosen Polinema ke: {OUT_CSV.name}")