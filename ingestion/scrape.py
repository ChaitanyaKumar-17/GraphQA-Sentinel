"""
ingestion/scrape.py

Crawls the target documentation site (FastAPI docs by default) and saves
the extracted content of each page as structured Markdown records.

Uses trafilatura for content extraction instead of a hardcoded CSS
selector, so this script is genuinely source-agnostic: point
TARGET_DOCS_URL at any docs site with a sitemap.xml and it should still
correctly strip nav/sidebar/footer chrome and keep just the article body.
"""

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

TARGET_DOCS_URL = os.getenv("TARGET_DOCS_URL", "https://fastapi.tiangolo.com/").rstrip("/") + "/"
OUTPUT_PATH = Path(__file__).parent / "data" / "raw" / "scraped_docs.json"
REQUEST_DELAY_SECONDS = 0.3
USER_AGENT = "GraphQA-Sentinel-Bot/1.0 (+educational project; polite crawler)"

# Language-prefixed URLs (translations) are excluded so we only ingest the
# canonical English docs.
LANG_PREFIX_RE = re.compile(r"^/[a-z]{2}(-[a-z]+)?/")


def get_sitemap_urls(base_url: str) -> list[str]:
    sitemap_url = base_url + "sitemap.xml"
    resp = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]


def is_english_docs_page(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    base_parsed = urlparse(base_url)
    if parsed.netloc != base_parsed.netloc:
        return False
    return not LANG_PREFIX_RE.match(parsed.path)


def extract_page_content(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [skip] {url} -> {e}")
        return None

    html = resp.text

    text = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )

    if not text or len(text) < 50:
        print(f"  [skip] {url} -> no extractable content")
        return None

    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else url

    return {"url": url, "title": title, "raw_text": text}


def main():
    print(f"Fetching sitemap for {TARGET_DOCS_URL} ...")
    all_urls = get_sitemap_urls(TARGET_DOCS_URL)
    doc_urls = [u for u in all_urls if is_english_docs_page(u, TARGET_DOCS_URL)]
    print(f"Found {len(all_urls)} total URLs, {len(doc_urls)} English docs pages to scrape.")

    pages = []
    for i, url in enumerate(doc_urls, start=1):
        print(f"[{i}/{len(doc_urls)}] {url}")
        page = extract_page_content(url)
        if page:
            pages.append(page)
        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(pages)} pages to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()