# 1. Turn off local SSL certificate warning spam in the terminal
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. Direct target link to the current issue page (Lightweight HTML)
url = "https://jurnal.polinema.ac.id/index.php/abdimas"

# Simulating a regular web browser to pass common security blocks
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("Connecting directly to Jurnal Informatika Polinema (Current Issue)... \n")

try:
    # Fetching the web page with a strict 20-second connection timeout limit
    response = requests.get(url, headers=headers, verify=False, timeout=20)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # In OJS 3 platforms, every paper is bundled inside a div with class 'obj_article_summary'
        articles = soup.find_all("div", class_="obj_article_summary")

        # Backup selector if the site is rendering a simplified layout variant
        if not articles:
            articles = soup.find_all("h4", class_="title") or soup.find_all("h3", class_="title")

        print(f"Successfully connected! Found {len(articles)} journal papers available.\n")

        # Let's extract data from the first 5 papers
        for i, article in enumerate(articles[:5]):
            # 1. Extract Paper Title
            title_node = article.find("h3", class_="title") or article.find("a")
            title = title_node.text.strip() if title_node else "No Title Available"

            # 2. Extract Authors
            author_node = article.find("div", class_="authors")
            authors = author_node.text.strip() if author_node else "Anonymous"

            # Remove excessive inner line breaks/tabs commonly added by OJS layouts
            authors = " ".join(authors.split())

            # 3. Extract PDF link if available
            pdf_node = article.find("a", class_="obj_galley_link pdf")
            pdf_link = pdf_node['href'] if pdf_node else "No PDF available directly"

            print(f"--- Paper No. {i+1} ---")
            print(f"Title   : {title}")
            print(f"Authors : {authors}")
            print(f"PDF Link: {pdf_link}\n")

    else:
        print(f"Could not read data. Server responded with HTTP Code: {response.status_code}")

except requests.exceptions.Timeout:
    print("Connection timed out: The university server took too long to load the page layout.")
except Exception as e:
    print(f"A structural issue occurred while parsing the text: {e}")