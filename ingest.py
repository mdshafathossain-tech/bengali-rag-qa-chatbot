"""
ingest.py
=========
Web scraper / ingestion script for a Bengali prose book Knowledge-Base RAG
Chatbot project.
 
WHAT THIS SCRIPT DOES
----------------------
1. Reads a list of chapter URLs (with custom chapter names you define below).
2. Downloads each page with `requests`.
3. Parses the HTML with `BeautifulSoup`, strips out navigation bars, headers,
   footers, scripts, ads, etc., and keeps only the "main story" text.
4. Attaches metadata (book_name, chapter_name, source_url, word_count,
   scraped_at) to each chapter.
5. Saves everything into a single structured `scraped_book.json` file that
   you can later chunk + embed for your RAG pipeline.
 
HOW TO ADJUST FOR YOUR OWN BOOK
--------------------------------
1. Change `BOOK_NAME` below to the actual title of your book.
2. Edit the `CHAPTERS` list — each entry is a dict:
       {"url": "<the page URL>", "chapter_name": "<your custom title>"}
   You can add as many chapters as you want, in any order.
3. Content extraction works by collecting every <p> (paragraph) tag left
   after junk removal — it does NOT depend on knowing the exact class name
   a site's theme uses for its content wrapper, so it works across most
   WordPress/blog-style sites out of the box. `CONTENT_SELECTORS` narrows
   the search to a specific container first (when one of those selectors
   matches), which helps on pages with multiple unrelated blocks of <p>
   text; if none match, it falls back to scanning the whole page. Add your
   own selector there only if you find the default extraction is pulling
   in <p> text from somewhere it shouldn't.
4. Run:  python ingest.py
   Output: scraped_book.json in the same folder.
 
REQUIREMENTS
------------
pip install requests beautifulsoup4
"""
 
import json
import re
import sys
import time
from datetime import datetime, timezone
 
import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — EDIT THIS SECTION FOR YOUR OWN BOOK
# ---------------------------------------------------------------------------

# Global book name (used as metadata for every chapter).
BOOK_NAME = "হাজার বছর ধরে"  # <-- change to your book's real title

# Flexible chapter configuration.
# Add / remove / reorder entries freely. `chapter_name` is whatever you want
# to show to the user later (it does NOT have to match anything on the page).
CHAPTERS = [
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%a7-%e0%a6%ae%e0%a6%b8%e0%a7%8d%e0%a6%a4-%e0%a6%ac%e0%a7%9c-%e0%a6%85%e0%a6%9c%e0%a6%97%e0%a6%b0%e0%a7%87%e0%a6%b0-%e0%a6%ae%e0%a6%a4-%e0%a6%b8%e0%a7%9c%e0%a6%95%e0%a6%9f%e0%a6%be/",
        "chapter_name": "০১. মস্ত বড় অজগরের মত সড়কটা",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%a8-%e0%a6%b0%e0%a6%be%e0%a6%a4%e0%a7%87%e0%a6%b0-%e0%a6%ac%e0%a7%87%e0%a6%b2%e0%a6%be-%e0%a6%86%e0%a6%ae%e0%a7%87%e0%a6%a8%e0%a6%be%e0%a6%b0-%e0%a6%98%e0%a6%b0%e0%a7%87-%e0%a6%b6/",
        "chapter_name": "০২. রাতের বেলা আমেনার ঘরে শোয় মকবুল",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%a9-%e0%a6%ac%e0%a6%be%e0%a7%9c%e0%a6%bf-%e0%a6%ab%e0%a6%bf%e0%a6%b0%e0%a7%87-%e0%a6%8f%e0%a6%b8%e0%a7%87-%e0%a6%ae%e0%a6%a8%e0%a7%8d%e0%a6%a4%e0%a7%81-%e0%a6%a6%e0%a7%87%e0%a6%96/",
        "chapter_name": "০৩. বাড়ি ফিরে এসে মন্তু দেখলো",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%aa-%e0%a6%98%e0%a7%81%e0%a6%ae-%e0%a6%ad%e0%a6%be%e0%a6%99%e0%a6%b2%e0%a7%8b-%e0%a6%95%e0%a6%96%e0%a6%a8/",
        "chapter_name": "০৪. ঘুম ভাঙলো কখন",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ab-%e0%a6%8f%e0%a6%96%e0%a6%a8-%e0%a6%a6%e0%a7%81%e0%a6%9f%e0%a7%8b-%e0%a6%96%e0%a7%81%e0%a6%81%e0%a7%9c%e0%a6%a4%e0%a7%87-%e0%a6%b9%e0%a6%ac%e0%a7%87/",
        "chapter_name": "০৫. এখন দুটো খুঁড়তে হবে",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ac-%e0%a6%ae%e0%a6%a8%e0%a7%8d%e0%a6%a4%e0%a7%81%e0%a6%b0-%e0%a6%ac%e0%a6%bf%e0%a7%9f%e0%a7%87%e0%a6%b0-%e0%a6%95%e0%a6%a5%e0%a6%be/",
        "chapter_name": "০৬. মঞ্জুর বিয়ের কথা",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ad-%e0%a6%86%e0%a6%ae%e0%a7%87%e0%a6%a8%e0%a6%be-%e0%a6%86%e0%a6%b0-%e0%a6%ab%e0%a6%be%e0%a6%a4%e0%a7%87%e0%a6%ae%e0%a6%be%e0%a6%b0-%e0%a6%ac%e0%a6%be%e0%a7%9c%e0%a6%bf-%e0%a6%a5/",
        "chapter_name": "০৭. আমেনা আর ফাতেমার বাড়ি থেকে",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%ae-%e0%a6%a6%e0%a6%bf%e0%a6%a8-%e0%a6%a4%e0%a6%bf%e0%a6%a8%e0%a7%87%e0%a6%95-%e0%a6%aa%e0%a6%b0/",
        "chapter_name": "০৮. দিন তিনেক পর",
    },
    {
        "url": "https://www.ebanglalibrary.com/lessons/%e0%a7%a6%e0%a7%af-%e0%a6%b9%e0%a6%be%e0%a6%9c%e0%a6%be%e0%a6%b0-%e0%a6%ac%e0%a6%9b%e0%a6%b0%e0%a7%87%e0%a6%b0-%e0%a6%aa%e0%a7%81%e0%a6%b0%e0%a6%a8%e0%a7%8b-%e0%a6%b8%e0%a7%87%e0%a6%87-%e0%a6%b0/",
        "chapter_name": "০৯. হাজার বছরের পুরনো সেই রাত",
    },
    # Add as many chapters/URLs as you like, e.g.:
    # {
    #     "url": "https://example.com/kapalkundala/chapter-3",
    #     "chapter_name": "অধ্যায় ৩ - বাটীতে প্রত্যাগমন",
    # },
]

# Output file
OUTPUT_FILE = "scraped_book.json"

# HTTP settings
REQUEST_TIMEOUT = 15          # seconds
RETRY_COUNT = 3                # retries per URL on failure
RETRY_BACKOFF_SECONDS = 2      # base backoff between retries
REQUEST_DELAY_SECONDS = 1      # polite delay between requests (be a good citizen)
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
 
# CSS selectors to try (in order) when looking for the "main story" content.
# The script uses the FIRST selector that yields substantial text.
# Add your own site-specific selector here if the default ones don't match.
CONTENT_SELECTORS = [
    "div.entry-content",   # common WordPress "lesson"/post body wrapper (e.g. ebanglalibrary.com)
    "div.post-content",
    "article",
    "main",
    "div.story-content",
    "div.content",
    "div#content",
    "div.article-body",
]
 
# Tags to strip out entirely before extracting text (nav bars, ads, etc.)
TAGS_TO_REMOVE = [
    "nav", "header", "footer", "script", "style", "noscript", "form",
    "aside", "iframe", "svg", "button", "input", "select", "label",
]
 
# Class/id keywords that usually indicate junk (ads, menus, sidebars, comments,
# login/registration modals, popups). LearnDash/LMS-style WordPress themes
# (like ebanglalibrary.com) render a login+register modal and a comment form
# on every lesson page, so "login", "register", "modal", "comment", and
# "respond" are included to strip those out too.
JUNK_KEYWORDS = [
    "advert", "ads", "sidebar", "menu", "nav", "footer", "header",
    "comment", "respond", "social", "share", "related", "breadcrumb",
    "widget", "cookie", "popup", "subscribe", "login", "register",
    "modal", "post-actions", "bookmark",
]
 
# Minimum characters required for a selector's extracted text to be
# considered "the real content" rather than a stray small block.
MIN_CONTENT_LENGTH = 200

# --- DIAGNOSTIC / DEBUG SETTINGS ---
# When True, the script saves the RAW HTML received for every chapter into
# DEBUG_DIR, and prints extra stats (raw <p> count, raw text length, etc.)
# to the console. Turn this on whenever extraction is failing and you can't
# tell whether the problem is (a) the site serving different content to a
# script than to a browser, (b) content not being in <p> tags, or (c) the
# junk-removal filter deleting too much. Turn it off once things work, to
# keep the console output clean and avoid writing files you don't need.
DEBUG_MODE = True
DEBUG_DIR = "debug_html"
 
 
# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
 
def log(message: str, level: str = "INFO") -> None:
    """Simple timestamped logger so progress is easy to follow in the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
 
 
def save_debug_html(html: str, chapter_index: int, chapter_name: str) -> None:
    """
    Save the raw HTML for a chapter to DEBUG_DIR so it can be inspected by
    hand. Only runs when DEBUG_MODE is True. Never raises — debugging output
    should never be the thing that crashes a run.
    """
    if not DEBUG_MODE:
        return
    try:
        import os
        os.makedirs(DEBUG_DIR, exist_ok=True)
        # Keep the filename simple/safe regardless of what characters are
        # in the chapter name (which may be Bengali, contain punctuation, etc.)
        safe_name = f"chapter_{chapter_index:02d}"
        path = os.path.join(DEBUG_DIR, f"{safe_name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"[DEBUG] Saved raw HTML for '{chapter_name}' -> {path}", level="INFO")
    except OSError as e:
        log(f"[DEBUG] Could not save debug HTML: {e}", level="WARNING")


def fetch_html(url: str) -> str | None:
    """
    Download the raw HTML for a URL, with retries and basic error handling.
    Returns the HTML string, or None if all attempts fail.
    """
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            log(f"Fetching (attempt {attempt}/{RETRY_COUNT}): {url}")
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            # Ensure correct decoding for Bengali/UTF-8 text
            response.encoding = response.apparent_encoding or "utf-8"

            if DEBUG_MODE:
                log(f"[DEBUG] HTTP {response.status_code}, "
                    f"{len(response.text)} chars received, "
                    f"final URL after redirects: {response.url}", level="INFO")

            return response.text

        except requests.exceptions.Timeout:
            log(f"Timeout while fetching {url}", level="WARNING")
        except requests.exceptions.HTTPError as e:
            log(f"HTTP error for {url}: {e}", level="WARNING")
        except requests.exceptions.RequestException as e:
            log(f"Request failed for {url}: {e}", level="WARNING")

        if attempt < RETRY_COUNT:
            sleep_time = RETRY_BACKOFF_SECONDS * attempt
            log(f"Retrying in {sleep_time}s...", level="INFO")
            time.sleep(sleep_time)

    log(f"Giving up on {url} after {RETRY_COUNT} attempts.", level="ERROR")
    return None
 
 
def is_junk_element(tag) -> bool:
    """Heuristic: does this tag's class/id look like nav/ad/sidebar/etc.?"""
    if getattr(tag, "decomposed", False):
        return False
    # NEVER treat <body> or <html> as junk. WordPress themes (like the
    # Genesis theme here) dump dozens of utility classes onto <body> for
    # CSS hooks — e.g. "no-sidebar" (meaning "this page has NO sidebar"),
    # "has-dark-header", "genesis-breadcrumbs-visible" — and those classes
    # legitimately contain words like "sidebar"/"header"/"breadcrumb" as
    # whole tokens even though the tag itself is the page root, not junk.
    # Decomposing <body> deletes the entire page. Structural root tags are
    # exempt from the keyword filter no matter what classes they carry.
    if tag.name in ("body", "html"):
        return False
    classes = tag.get("class") or []
    tag_id = tag.get("id") or ""
    identifiers = (" ".join(classes) + " " + tag_id).lower()
    return any(keyword in identifiers for keyword in JUNK_KEYWORDS)
 
 
def clean_soup(soup: BeautifulSoup) -> None:
    """
    Mutates `soup` in place: removes script/style/nav/footer/etc. tags,
    HTML comments, and elements whose class/id suggests junk content
    (ads, menus, sidebars, comment sections...).
    """
    # Remove structural junk tags entirely
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            if getattr(tag, "decomposed", False):
                continue
            if DEBUG_MODE:
                _log_big_removal(tag, reason=f"<{tag_name}> tag")
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove elements that look like ads/menus/sidebars/comments by class or id.
    # We re-check `.decomposed` on every tag here too, because decomposing
    # one element in this same loop can invalidate its still-queued children
    # (see the note in is_junk_element above).
    for tag in soup.find_all(True):
        if getattr(tag, "decomposed", False):
            continue
        if is_junk_element(tag):
            if DEBUG_MODE:
                classes = tag.get("class") or []
                tag_id = tag.get("id") or ""
                matched = [kw for kw in JUNK_KEYWORDS
                           if kw in (" ".join(classes) + " " + tag_id).lower()]
                _log_big_removal(
                    tag,
                    reason=f"class/id keyword match {matched} "
                           f"(class={classes!r}, id={tag_id!r})",
                )
            tag.decompose()


def _log_big_removal(tag, reason: str) -> None:
    """
    DEBUG-only helper: if a tag we're about to decompose contains a
    suspiciously large amount of text (>= 500 chars), log it loudly.
    This is how we catch a junk filter accidentally deleting the real
    article content along with an ancestor wrapper.
    """
    try:
        text_len = len(tag.get_text(strip=True))
    except Exception:  # noqa: BLE001 - diagnostics must never crash the run
        text_len = -1
    if text_len >= 500:
        log(f"[DEBUG] !!! Removing <{tag.name}> with {text_len} chars of text "
            f"due to: {reason} — THIS MAY BE DELETING REAL CONTENT.",
            level="WARNING")
 
 
def extract_paragraphs(container) -> str:
    """
    Collect the text of every <p> tag inside `container` and join them with
    newlines, one paragraph per line.

    WHY PARAGRAPH-BASED EXTRACTION:
    Story/article text in almost every WordPress-style site lives inside
    <p> tags. Chapter navigation links, breadcrumb link lists, login/register
    modal fields (<label>/<input>), and social-share buttons generally are
    NOT wrapped in <p> tags. So instead of trying to guess the exact class
    name a given theme uses for its content wrapper (which varies site to
    site and breaks silently when it doesn't match), we just grab every
    surviving <p> after junk removal. This is far more robust across
    different source websites.
    """
    paragraphs = []
    for p in container.find_all("p"):
        # Guard against tags decomposed as a side effect of an ancestor's
        # removal (see the note in is_junk_element / clean_soup above).
        if getattr(p, "decomposed", False):
            continue
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_main_text(html: str) -> str:
    """
    Parse HTML and extract the main story text, stripped of navigation,
    headers, footers, ads, login/comment forms, and other boilerplate.

    Strategy (each step only runs if the previous one came up short):
      1. Try each selector in CONTENT_SELECTORS, extracting only the <p>
         tags found inside it (not its raw get_text(), which could still
         include stray link-list or heading text near the content).
      2. If no selector matched well, collect every <p> in the whole
         (already-cleaned) document. This is the most robust fallback and
         is often enough on its own for typical article/story pages.
      3. As an absolute last resort (some very simple pages don't wrap
         text in <p> at all), fall back to the full cleaned <body> text.
    """
    raw_soup_for_diagnostics = None
    if DEBUG_MODE:
        # Parse a SEPARATE, un-cleaned copy purely to report "before" stats —
        # this never affects the real extraction below.
        raw_soup_for_diagnostics = BeautifulSoup(html, "html.parser")

    soup = BeautifulSoup(html, "html.parser")

    if DEBUG_MODE and raw_soup_for_diagnostics is not None:
        raw_body = raw_soup_for_diagnostics.find("body")
        raw_p_count = len(raw_soup_for_diagnostics.find_all("p"))
        raw_body_text_len = len(raw_body.get_text(strip=True)) if raw_body else 0
        log(f"[DEBUG] BEFORE cleaning: {raw_p_count} <p> tags, "
            f"{raw_body_text_len} chars of body text.", level="INFO")

    clean_soup(soup)

    if DEBUG_MODE:
        clean_body = soup.find("body")
        clean_p_count = len(soup.find_all("p"))
        clean_body_text_len = len(clean_body.get_text(strip=True)) if clean_body else 0
        log(f"[DEBUG] AFTER cleaning: {clean_p_count} <p> tags, "
            f"{clean_body_text_len} chars of body text remain.", level="INFO")

    best_text = ""

    # Step 1: try each candidate selector, using only its <p> tags
    for selector in CONTENT_SELECTORS:
        element = soup.select_one(selector)
        if element:
            text = extract_paragraphs(element)
            if len(text) >= MIN_CONTENT_LENGTH:
                best_text = text
                break
            # Keep it as a fallback candidate even if short, in case nothing better appears
            if len(text) > len(best_text):
                best_text = text

    # Step 2: no selector worked well — gather every <p> in the whole page
    if len(best_text) < MIN_CONTENT_LENGTH:
        body = soup.find("body")
        if body:
            whole_page_text = extract_paragraphs(body)
            if len(whole_page_text) > len(best_text):
                best_text = whole_page_text

    # Step 3: still nothing — fall back to raw cleaned <body> text
    if len(best_text) < MIN_CONTENT_LENGTH:
        body = soup.find("body")
        if body:
            raw_text = body.get_text(separator="\n", strip=True)
            if len(raw_text) > len(best_text):
                best_text = raw_text

    return normalize_text(best_text)
 
 
def normalize_text(text: str) -> str:
    """Collapse excessive blank lines/whitespace while preserving paragraphs."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    cleaned = "\n".join(lines)
    # Collapse 3+ blank-line-equivalents / repeated spaces
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
 
 
def scrape_chapter(chapter_config: dict, chapter_index: int = 0) -> dict | None:
    """
    Scrape a single chapter given its config dict:
        {"url": ..., "chapter_name": ...}
    Returns a dict with metadata + content, or None if scraping failed.
    """
    url = chapter_config.get("url")
    chapter_name = chapter_config.get("chapter_name", "Unknown Chapter")
 
    if not url:
        log(f"Skipping entry with missing URL: {chapter_config}", level="ERROR")
        return None
 
    html = fetch_html(url)
    if html is None:
        return None

    if DEBUG_MODE:
        save_debug_html(html, chapter_index, chapter_name)
 
    try:
        content = extract_main_text(html)
    except Exception as e:  # noqa: BLE001 - we want to log *any* parsing failure
        log(f"Failed to parse content for {url}: {e}", level="ERROR")
        return None
 
    if not content:
        log(f"No content extracted for '{chapter_name}' ({url}).", level="WARNING")
        return None
 
    word_count = len(content.split())
    log(f"Extracted '{chapter_name}' — {word_count} words.", level="SUCCESS")
 
    return {
        "book_name": BOOK_NAME,
        "chapter_name": chapter_name,
        "source_url": url,
        "word_count": word_count,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
    }
 
 
# ---------------------------------------------------------------------------
# 3. MAIN PIPELINE
# ---------------------------------------------------------------------------
 
def run_ingestion(chapters: list[dict]) -> list[dict]:
    """Scrape every chapter in `chapters`, skipping failures, return results list."""
    results = []
    total = len(chapters)
 
    if total == 0:
        log("CHAPTERS list is empty — nothing to scrape. Edit the CONFIGURATION section.",
            level="ERROR")
        return results
 
    for index, chapter_config in enumerate(chapters, start=1):
        log(f"--- Processing chapter {index}/{total} ---")
        try:
            chapter_data = scrape_chapter(chapter_config, chapter_index=index)
            if chapter_data:
                results.append(chapter_data)
            else:
                log(f"Chapter {index}/{total} skipped due to errors.", level="WARNING")
        except Exception as e:  # noqa: BLE001 - top-level safety net per chapter
            log(f"Unexpected error on chapter {index}/{total}: {e}", level="ERROR")
 
        # Be polite to the server between requests
        if index < total:
            time.sleep(REQUEST_DELAY_SECONDS)
 
    return results
 
 
def save_results(results: list[dict], output_path: str) -> None:
    """Write the scraped data to a JSON file with nice formatting."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log(f"Saved {len(results)} chapter(s) to '{output_path}'.", level="SUCCESS")
    except OSError as e:
        log(f"Failed to write output file '{output_path}': {e}", level="ERROR")
        sys.exit(1)
 
 
def main() -> None:
    log(f"Starting ingestion for book: '{BOOK_NAME}'")
    log(f"Total chapters configured: {len(CHAPTERS)}")
 
    results = run_ingestion(CHAPTERS)
 
    if not results:
        log("No chapters were successfully scraped. Exiting without writing output.",
            level="ERROR")
        sys.exit(1)
 
    save_results(results, OUTPUT_FILE)
 
    total_words = sum(r["word_count"] for r in results)
    log(f"Done. {len(results)}/{len(CHAPTERS)} chapters scraped successfully "
        f"({total_words} total words).", level="SUCCESS")
 
 
if __name__ == "__main__":
    main()