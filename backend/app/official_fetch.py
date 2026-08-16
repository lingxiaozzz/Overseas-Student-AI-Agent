from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import aiohttp

from app.config import settings


ALLOWED_HOST_SUFFIXES = (
    "homeaffairs.gov.au",
    "studyaustralia.gov.au",
    "sydney.edu.au",
    "privatehealth.gov.au",
)

OFFICIAL_PAGES = (
    {
        "topics": ("visa", "student visa", "subclass 500", "immi", "home affairs"),
        "url": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500",
        "title": "Student visa (subclass 500) - Home Affairs",
    },
    {
        "topics": ("oshc", "health cover", "health insurance", "overseas student health"),
        "url": "https://www.studyaustralia.gov.au/en/plan-your-move/overseas-student-health-cover-oshc",
        "title": "Overseas Student Health Cover - Study Australia",
    },
    {
        "topics": ("oshc", "health cover", "bupa"),
        "url": "https://www.sydney.edu.au/students/overseas-student-health-cover-oshc.html",
        "title": "OSHC - University of Sydney",
    },
    {
        "topics": ("usyd", "university of sydney", "international student", "enrol", "orientation"),
        "url": "https://www.sydney.edu.au/study/why-choose-sydney/international-students.html",
        "title": "International students - University of Sydney",
    },
    {
        "topics": ("accommodation", "housing", "rent", "campus living"),
        "url": "https://www.sydney.edu.au/study/accommodation.html",
        "title": "Accommodation - University of Sydney",
    },
    {
        "topics": ("oshc", "private health"),
        "url": "https://www.privatehealth.gov.au/health_insurance/overseas/overseas_student_health_cover.htm",
        "title": "OSHC - PrivateHealth.gov.au",
    },
)


class OfficialFetchError(ValueError):
    pass


@dataclass
class OfficialPage:
    url: str
    title: str
    text: str


class _HtmlTextExtractor(HTMLParser):
    _skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title = ""
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
            return
        if self._skip_depth:
            return
        self._chunks.append(text)
        self._chunks.append(" ")

    def text(self) -> str:
        compact = " ".join("".join(self._chunks).split())
        return compact


def is_allowed_official_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in ALLOWED_HOST_SUFFIXES)


def list_official_sources() -> list[dict[str, str]]:
    seen: set[str] = set()
    pages: list[dict[str, str]] = []
    for item in OFFICIAL_PAGES:
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        pages.append({"title": str(item["title"]), "url": url})
    return pages


def suggest_official_urls(query: str, *, limit: int | None = None) -> list[str]:
    max_pages = limit if limit is not None else settings.official_fetch_max_pages
    lowered = query.lower()
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for item in OFFICIAL_PAGES:
        url = str(item["url"])
        if url in seen:
            continue
        score = sum(1 for topic in item["topics"] if topic in lowered)
        if score <= 0:
            continue
        seen.add(url)
        ranked.append((score, url))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked and any(token in lowered for token in ("usyd", "sydney", "visa", "oshc", "arrival")):
        ranked = [(1, str(OFFICIAL_PAGES[0]["url"]))]
    return [url for _score, url in ranked[: max(0, max_pages)]]


def _extract_html_text(html: str) -> tuple[str, str]:
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text()


async def fetch_official_page(url: str) -> OfficialPage:
    if not is_allowed_official_url(url):
        raise OfficialFetchError(
            "URL is not on the official allowlist "
            "(homeaffairs.gov.au, studyaustralia.gov.au, sydney.edu.au, privatehealth.gov.au)."
        )

    timeout = aiohttp.ClientTimeout(total=settings.official_fetch_timeout_seconds)
    headers = {
        "User-Agent": "Overseas-Student-AI-Agent/1.0 (educational official-source fetch)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(url, allow_redirects=True) as response:
            final_url = str(response.url)
            if not is_allowed_official_url(final_url):
                raise OfficialFetchError("Redirect left the official allowlist.")
            if response.status >= 400:
                raise OfficialFetchError(f"Official page returned HTTP {response.status}.")
            html = await response.text(errors="replace")

    title, text = _extract_html_text(html)
    max_chars = max(200, settings.official_fetch_max_chars)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return OfficialPage(url=final_url, title=title or final_url, text=text or "No readable text extracted.")


async def fetch_official_pages_for_query(query: str) -> list[OfficialPage]:
    if not settings.official_fetch_enabled:
        return []
    pages: list[OfficialPage] = []
    for url in suggest_official_urls(query):
        try:
            pages.append(await fetch_official_page(url))
        except Exception:
            continue
    return pages
