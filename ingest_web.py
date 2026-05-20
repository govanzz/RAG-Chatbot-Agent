"""Scrape configured Mandai URLs and build the local FAISS vectorstore."""

from __future__ import annotations

import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from config import (
    INGEST_LOG_PATH,
    MANDAI_URLS,
    RAW_PAGES_DIR,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT,
    USER_AGENT,
    VECTORSTORE_DIR,
    LOGS_DIR,
)
from rag import build_vectorstore
from utils import (
    append_jsonl,
    clean_inline_text,
    dedupe_preserve_order,
    ensure_directories,
    page_filename,
    utc_now_iso,
)

REMOVE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "[aria-hidden='true']",
    ".breadcrumb",
    ".breadcrumbs",
    ".cookie",
    ".modal",
]

BOILERPLATE_TEXT = {
    "menu",
    "search",
    "close",
    "open",
    "skip to main content",
    "book tickets",
    "buy tickets",
    "sign in",
    "login",
}


def _remove_noise(soup: BeautifulSoup) -> None:
    for selector in REMOVE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()


def _extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        title = clean_inline_text(h1.get_text(" ", strip=True))
        if title:
            return title

    if soup.title:
        title = clean_inline_text(soup.title.get_text(" ", strip=True))
        if title:
            return title

    return "Mandai page"


def _extract_content(soup: BeautifulSoup) -> tuple[list[str], str]:
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    headings: list[str] = []
    last_line = ""

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean_inline_text(element.get_text(" ", strip=True))
        if not text or text.casefold() in BOILERPLATE_TEXT:
            continue

        if element.name in {"h1", "h2", "h3", "h4"}:
            level = int(element.name[1])
            line = f"{'#' * min(level, 4)} {text}"
            headings.append(text)
        elif element.name == "li":
            line = f"- {text}"
        else:
            line = text

        if line != last_line:
            lines.append(line)
            last_line = line

    body = "\n".join(lines).strip()
    return dedupe_preserve_order(headings), body


def scrape_page(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    _remove_noise(soup)
    title = _extract_title(soup)
    headings, body = _extract_content(soup)

    if not body:
        raise RuntimeError("No main page text could be extracted.")

    output_path = RAW_PAGES_DIR / page_filename(url)
    document = "\n".join(
        [
            f"Page Title: {title}",
            f"Source URL: {url}",
            f"Scraped At: {utc_now_iso()}",
            "Headings:",
            *[f"- {heading}" for heading in headings],
            "---",
            body,
            "",
        ]
    )
    output_path.write_text(document, encoding="utf-8")

    return {
        "url": url,
        "title": title,
        "path": str(output_path),
        "characters": len(body),
    }


def scrape_configured_urls() -> list[dict[str, Any]]:
    ensure_directories([RAW_PAGES_DIR, VECTORSTORE_DIR, LOGS_DIR])
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    scraped: list[dict[str, Any]] = []
    for index, url in enumerate(MANDAI_URLS, start=1):
        print(f"[{index}/{len(MANDAI_URLS)}] Scraping {url}")
        try:
            page = scrape_page(session, url)
            scraped.append(page)
            print(f"  Saved {page['path']} ({page['characters']} characters)")
        except Exception as exc:  # Continue politely through page-level failures.
            print(f"  Failed: {exc}")
            append_jsonl(
                INGEST_LOG_PATH,
                {
                    "timestamp": utc_now_iso(),
                    "url": url,
                    "error": str(exc),
                },
            )

        if index < len(MANDAI_URLS):
            time.sleep(REQUEST_DELAY_SECONDS)

    return scraped


def main() -> None:
    scraped = scrape_configured_urls()
    if not scraped and not list(RAW_PAGES_DIR.glob("*.txt")):
        raise SystemExit("No pages were scraped. Check logs/ingest_failures.jsonl.")

    print("Building FAISS vectorstore from scraped pages...")
    chunk_count = build_vectorstore(RAW_PAGES_DIR)
    print(f"Done. Indexed {chunk_count} chunks in vectorstore/.")


if __name__ == "__main__":
    main()

