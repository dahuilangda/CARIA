"""Literature discovery and full-text retrieval for the extraction harness.

Finds open-access nanocarrier articles and fetches their full text so the
extraction harness can process them. Three stages, each independently
callable and each returning plain data:

search
    PubMed queries via ``pymed`` plus Tavily web search (``TAVILY_API_KEY``)
    as a complementary discovery channel. Tavily quota errors degrade to
    "skipped channel", never abort the run.

link
    PMID -> PMC ID via the NCBI elink API.

fetch
    PMC ID -> full-text XML via the Europe PMC REST endpoint (the legal
    open-access path used by the corpus).

Every fetched article is cached on disk as ``<pmcid>.xml``; repeated runs are
idempotent and never re-download.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

PUBMED_TOOL = "CARIA"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
REQUEST_TIMEOUT = 60.0
RATE_LIMIT_SECONDS = 0.4  # NCBI allows <=3 requests/second without a key


@dataclass
class Article:
    pmid: str
    title: str
    doi: str | None = None
    pmcid: str | None = None
    source: str = "pubmed"


@dataclass
class FetchResult:
    pmcid: str
    path: Path | None = None
    status: str = "pending"  # fetched | cached | no_pmc | http_error | empty
    detail: str = ""


# ---------------------------------------------------------------- search
def search_pubmed(query: str, max_results: int = 50,
                  email: str = "caria@example.org") -> list[Article]:
    """Run one PubMed query and return deduplicated articles with a PMID."""
    from pymed import PubMed
    client = PubMed(tool=PUBMED_TOOL, email=email)
    articles: list[Article] = []
    seen: set[str] = set()
    for record in client.query(query, max_results=max_results):
        pmid = str(record.pubmed_id or "").split("\n")[0].strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        articles.append(Article(pmid=pmid, title=str(record.title or ""), doi=record.doi))
    return articles


class TavilyUnavailable(RuntimeError):
    """Raised when the Tavily channel cannot run (quota, auth, or missing SDK)."""


def search_tavily(query: str, max_results: int = 20,
                  api_key: str | None = None) -> list[Article]:
    """Discover articles through Tavily web search.

    Returns articles whose URL encodes a PMID (ncbi.nlm.nih.gov/pubmed/...).
    Quota or authentication failures raise :class:`TavilyUnavailable` so the
    caller can skip the channel and continue.
    """
    try:
        from tavily import TavilyClient
    except ImportError as error:
        raise TavilyUnavailable("tavily-python is not installed") from error
    key = api_key or _setting("TAVILY_API_KEY")
    if not key:
        raise TavilyUnavailable("TAVILY_API_KEY is not configured")
    try:
        response = TavilyClient(api_key=key).search(query, max_results=max_results)
    except Exception as error:  # quota/auth/network all degrade the same way
        raise TavilyUnavailable(f"Tavily search unavailable: {error}") from error
    import re
    articles: list[Article] = []
    for item in response.get("results", []):
        match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", item.get("url", ""))
        if match:
            articles.append(Article(pmid=match.group(1), title=item.get("title", ""),
                                    source="tavily"))
    return articles


def _setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                return line.partition("=")[2].strip().strip("'\"")
    return None


# ---------------------------------------------------------------- link
def pmid_to_pmcid(pmid: str) -> str | None:
    """Resolve one PMID to its PMC ID via NCBI elink; None when not open-access."""
    url = f"{EUTILS}/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json"
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
        data = json.loads(response.read().decode("utf-8"))
    for linkset in data.get("linksets", []):
        for db in linkset.get("linksetdbs", []):
            if db.get("dbto") == "pmc" and db.get("links"):
                return f"PMC{db['links'][0]}"
    return None


# ---------------------------------------------------------------- fetch
def fetch_fulltext_xml(pmcid: str, cache_dir: Path) -> FetchResult:
    """Download one PMC full-text XML into `cache_dir`, skipping cached files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{pmcid}.xml"
    if target.is_file() and target.stat().st_size > 0:
        return FetchResult(pmcid=pmcid, path=target, status="cached")
    url = f"{EUROPE_PMC}/{pmcid}/fullTextXML"
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        return FetchResult(pmcid=pmcid, status="http_error", detail=f"HTTP {error.code}")
    except urllib.error.URLError as error:
        return FetchResult(pmcid=pmcid, status="http_error", detail=str(error.reason))
    if not payload:
        return FetchResult(pmcid=pmcid, status="empty")
    target.write_bytes(payload)
    return FetchResult(pmcid=pmcid, path=target, status="fetched")


PDF_URLS = (
    "{EUROPE_PMC}/{pmcid}.pdf",
    "https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
)


def fetch_pdf(pmcid: str, cache_dir: Path) -> FetchResult:
    """Download one PMC article PDF into `cache_dir`, skipping cached files.

    PDFs keep the figures that XML lacks, letting vision-capable extraction
    models read the tables and plots directly. Several URL patterns are tried
    in order; many archives restrict programmatic PDF access, in which case
    the XML fetch (always available for the open-access subset) carries the
    text and the failure is recorded in the manifest.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{pmcid}.pdf"
    if target.is_file() and target.stat().st_size > 0:
        return FetchResult(pmcid=pmcid, path=target, status="cached")
    last_detail = ""
    for pattern in PDF_URLS:
        url = pattern.format(EUROPE_PMC=EUROPE_PMC, pmcid=pmcid)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT * 2) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            last_detail = f"HTTP {error.code}"
            continue
        except urllib.error.URLError as error:
            last_detail = str(error.reason)
            continue
        if payload.startswith(b"%PDF"):
            target.write_bytes(payload)
            return FetchResult(pmcid=pmcid, path=target, status="fetched")
        last_detail = "no PDF payload"
    return FetchResult(pmcid=pmcid, status="http_error", detail=last_detail)


def extract_figures_from_xml(xml_path: Path, out_dir: Path) -> list[Path]:
    """Extract embedded figure images referenced by one article XML.

    Open-access article XML links its figures as separate image files under
    the same article directory on Europe PMC. This helper rewrites those
    relative hrefs to Europe PMC image URLs, downloads them into `out_dir`
    and returns the local paths - giving vision-capable extraction models
    the figures even when PDF download is refused.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_text = Path(xml_path).read_text(encoding="utf-8", errors="replace")
    import re
    hrefs = re.findall(r'xlink:href="([^"]+\.(?:jpg|jpeg|gif|tif|tiff|png))"', xml_text, re.I)
    saved: list[Path] = []
    for href in dict.fromkeys(hrefs):
        name = href.split("/")[-1]
        url = f"{EUROPE_PMC}/{xml_path.stem}/{name}"
        target = out_dir / name
        if target.is_file() and target.stat().st_size > 0:
            saved.append(target)
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        if payload:
            target.write_bytes(payload)
            saved.append(target)
    return saved


def pmcid_from_xml(xml_text: str) -> str | None:
    """Read the PMC ID from an article XML document when present."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for front in root.iter("front"):
        for article_id in front.iter("article-id"):
            if article_id.get("pub-id-type") == "pmc":
                text = (article_id.text or "").strip()
                return text if text.upper().startswith("PMC") else f"PMC{text}"
    return None


def collect(query: str, cache_dir: Path, max_results: int = 50,
            tavily_key: str | None = None, email: str = "caria@example.org") -> dict:
    """Run search -> link -> fetch for one query and return the run ledger."""
    articles = search_pubmed(query, max_results=max_results, email=email)
    try:
        articles += [a for a in search_tavily(query, max_results=max_results,
                                              api_key=tavily_key)
                     if a.pmid not in {b.pmid for b in articles}]
        tavily_status = "ok"
    except TavilyUnavailable as error:
        tavily_status = str(error)
    unique: dict[str, Article] = {}
    for article in articles:
        unique.setdefault(article.pmid, article)
    manifest = {"query": query, "pubmed_hits": len(articles), "unique": len(unique),
                "tavily": tavily_status, "articles": []}
    cache = Path(cache_dir)
    for pmid, article in sorted(unique.items()):
        time.sleep(RATE_LIMIT_SECONDS)
        pmcid = pmid_to_pmcid(pmid)
        entry = {"pmid": pmid, "title": article.title, "source": article.source,
                 "pmcid": pmcid}
        if pmcid:
            result = fetch_fulltext_xml(pmcid, cache)
            entry.update({"fetch_status": result.status, "xml_path": str(result.path or "")})
            pdf = fetch_pdf(pmcid, cache)
            entry.update({"pdf_status": pdf.status, "pdf_path": str(pdf.path or "")})
        else:
            entry["fetch_status"] = "no_pmc"
        manifest["articles"].append(entry)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search PubMed/Tavily and cache PMC full texts.")
    parser.add_argument("query")
    parser.add_argument("--cache-dir", type=Path, default=Path("fulltext_xml"))
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--manifest", type=Path, default=Path("collection_manifest.json"))
    args = parser.parse_args()
    ledger = collect(args.query, args.cache_dir, args.max_results,
                     tavily_key=_setting("TAVILY_API_KEY"))
    Path(args.manifest).write_text(json.dumps(ledger, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in ledger.items() if k != "articles"}, indent=1))
