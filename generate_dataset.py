import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import time

nltk.download("punkt_tab")

BASE_URL = "https://en.wikipedia.org"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (Academic Project)"}

MAX_SENTENCES = 25000
MAX_KEYWORDS = 500


# --------------------------------------------------
# 1. Collect keywords from Wikipedia categories
# --------------------------------------------------

def get_keywords_from_category(category_link, max_terms=500):
    collected_terms = set()
    current_page = category_link

    while current_page and len(collected_terms) < max_terms:
        print(f"Scanning category page: {current_page}")

        response = requests.get(current_page, headers=USER_AGENT)
        soup = BeautifulSoup(response.text, "html.parser")

        entries = soup.select(".mw-category a")
        for entry in entries:
            word = entry.get_text().strip().lower()
            if len(word) > 3:
                collected_terms.add(word)

        next_button = soup.find("a", string="next page")
        if next_button:
            current_page = BASE_URL + next_button["href"]
        else:
            current_page = None

        time.sleep(0.5)

    return list(collected_terms)[:max_terms]


# --------------------------------------------------
# 2. Validate article titles
# --------------------------------------------------

def title_is_relevant(title, keyword_list):
    title_check = title.lower()

    if ":" in title_check:
        return False

    if "disambiguation" in title_check:
        return False

    for term in keyword_list:
        if term in title_check:
            return True

    return False


# --------------------------------------------------
# 3. Extract article content
# --------------------------------------------------

def scrape_article(article_title):
    full_url = f"{BASE_URL}/wiki/{article_title}"

    try:
        response = requests.get(full_url, headers=USER_AGENT)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraph_tags = soup.find_all("p")
        combined_text = " ".join(p.get_text() for p in paragraph_tags)

        outgoing_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/wiki/") and ":" not in href:
                outgoing_links.append(href.replace("/wiki/", ""))

        return combined_text, outgoing_links

    except Exception:
        return "", []


# --------------------------------------------------
# 4. Sentence crawling and labeling
# --------------------------------------------------

def crawl_sentences(seed_list, keywords, class_label):
    seen_pages = set()
    to_visit = list(seed_list)
    dataset = []

    while to_visit and len(dataset) < MAX_SENTENCES:
        page_title = to_visit.pop(0)

        if page_title in seen_pages:
            continue

        seen_pages.add(page_title)

        if not title_is_relevant(page_title, keywords):
            continue

        print(f"Downloading article: {page_title}")

        article_text, related_links = scrape_article(page_title)
        sentence_list = nltk.sent_tokenize(article_text)

        for sent in sentence_list:
            if len(sent.split()) > 8:
                dataset.append((sent, class_label))

            if len(dataset) >= MAX_SENTENCES:
                break

        for link in related_links:
            if link not in seen_pages and title_is_relevant(link, keywords):
                to_visit.append(link)

        time.sleep(0.3)

    return dataset


# --------------------------------------------------
# Main Execution
# --------------------------------------------------

def run_pipeline():
    print("\nExtracting Sports Keywords...")
    sports_terms = get_keywords_from_category(
        "https://en.wikipedia.org/wiki/Category:Sports",
        MAX_KEYWORDS
    )

    print("\nExtracting Politics Keywords...")
    politics_terms = get_keywords_from_category(
        "https://en.wikipedia.org/wiki/Category:Politics",
        MAX_KEYWORDS
    )

    print("\nGathering Sports Sentences...")
    sports_samples = crawl_sentences(["Sports"], sports_terms, 0)

    print("\nGathering Politics Sentences...")
    politics_samples = crawl_sentences(["Politics"], politics_terms, 1)

    full_dataset = sports_samples + politics_samples
    df = pd.DataFrame(full_dataset, columns=["sentence", "label"])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("dataset_title_filtered.csv", index=False)

    print("\nDataset generation complete.")
    print("Sports samples:", len(sports_samples))
    print("Politics samples:", len(politics_samples))
    print("Total samples:", len(df))


if __name__ == "__main__":
    run_pipeline()
