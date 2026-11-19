from pathlib import Path
import requests, urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
SINTA_URL = "https://sinta.kemdiktisaintek.go.id/affiliations/authors/413"

# test connection 
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36"}
resp = requests.get(SINTA_URL, params={"page": 1}, headers=HEADERS, verify=False, timeout=25)
soup = BeautifulSoup(resp.text, "html.parser")
author_cards = soup.find_all("div", class_="au-item")
print(f"Status HTTP: {resp.status_code} | Jumlah Dosen di Halaman 1: {len(author_cards)}")

# test liat struktur html 
for tag in author_cards[0].find_all(['div', 'a', 'span'], class_=True):
    print(f'{tag.name:<5} | class={tag.get('class')} | text={' '.join(tag.get_text().split())[:70]}')
    