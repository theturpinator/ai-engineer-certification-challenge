"""Ingestion: cleaned article CSV -> committed index artifact.

Run from api/ with the venv set up (see README.md):

    uv run python -m ingest

Reads ../data/articles-clean.csv, strips Webflow HTML, chunks along authored
block boundaries (intro paragraph + copy blocks), embeds each chunk with
text-embedding-3-small via the Vercel AI Gateway, and writes:

    index_artifact/chunks.jsonl   one chunk record (text + metadata) per line
    index_artifact/vectors.npz    float32 vectors, row i matches chunks line i

The 21 expired sweepstakes/promo articles (see EXCLUDED_ARTICLES.md) are
excluded by slug.
"""

import csv
import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import tiktoken

API_DIR = Path(__file__).parent
CSV_PATH = API_DIR.parent / "data" / "articles-clean.csv"
ARTIFACT_DIR = API_DIR / "index_artifact"

BLOCK_COLUMNS = ["INTRO PARAGRAPH", "1 copy block", "2 copy block",
                 "3 copy block", "4 copy block"]
MAX_TOKENS = 1000
OVERLAP_TOKENS = 100
MIN_CHUNK_CHARS = 200

# Expired sweepstakes/giveaway promos — rule and full titles in EXCLUDED_ARTICLES.md.
EXCLUDED_SLUGS = frozenset({
    "1970-boss-mustang",
    "5-ways-you-can-win-a-2020-ford-mustang-gt500",
    "cruise-for-a-cause-sweepstakes-shelby-gt500se",
    "customized-convenience",
    "drive-home-hope",
    "drive-the-dream",
    "fever-dream",
    "get-ready",
    "jdrf-2024-mustang-dark-horse",
    "prize-pony",
    "roush-raffle-support-the-preservation-of-henry-fords-home-and-you-might-win-a-special-roush-mustang",
    "shelby-gt500-sweepstakes-helps-veterans-in-need",
    "shelby-sweepstakes-win-an-800hp-2021-shelby-gt500se",
    "ten-for-the-win-more-chances-to-win-great-prizes",
    "win-a-1968-shelby-gt500",
    "win-a-2021-mustang-gt-and-ford-performance-parts",
    "win-a-mustang-mach-e-gt-charging-station",
    "win-a-mustang-mach-e-gt-help-joey-loganos-charity",
    "win-a-one-of-a-kind-shelby-snakecharmer-for-just-25",
    "win-an-800hp-saleen-302-for-as-little-as-25",
    "win-this-1969-mach-1-mustang",
})

_enc = tiktoken.get_encoding("cl100k_base")


def _n_tokens(text: str) -> int:
    return len(_enc.encode(text))


class _TextExtractor(HTMLParser):
    """Collects text; block-level closing tags become newlines."""

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div",
                  "blockquote", "figure", "figcaption", "br", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")


def strip_html(html: str) -> str:
    """Webflow rich-text HTML -> plain text with newline-separated blocks."""
    parser = _TextExtractor()
    parser.feed(html or "")
    text = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def split_oversize(text: str) -> list[str]:
    """Recursively split text over MAX_TOKENS at the midpoint with overlap."""
    tokens = _enc.encode(text)
    if len(tokens) <= MAX_TOKENS:
        return [text]
    mid = len(tokens) // 2
    half = OVERLAP_TOKENS // 2
    left = _enc.decode(tokens[: mid + half])
    right = _enc.decode(tokens[mid - half:])
    return split_oversize(left) + split_oversize(right)


def merge_tiny(pieces: list[str]) -> list[str]:
    """Merge fragments under MIN_CHUNK_CHARS into the previous (or next) piece."""
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n\n" + piece
        else:
            merged.append(piece)
    if len(merged) >= 2 and len(merged[-1]) < MIN_CHUNK_CHARS:
        tail = merged.pop()
        merged[-1] = merged[-1] + "\n\n" + tail
    return merged


def _parse_published(raw: str) -> str | None:
    # e.g. "Tue Sep 14 2021 11:32:05 GMT+0000 (Coordinated Universal Time)"
    m = re.match(r"\w{3} (\w{3} \d{2} \d{4})", raw or "")
    return datetime.strptime(m.group(1), "%b %d %Y").date().isoformat() if m else None


def chunk_article(row: dict) -> list[dict]:
    """One CSV row -> chunk records (title prepended to each chunk's text)."""
    title = row["Title"].strip()
    bodies: list[str] = []
    for col in BLOCK_COLUMNS:
        block = strip_html(row.get(col, ""))
        if block:
            bodies.extend(split_oversize(block))
    bodies = merge_tiny(bodies)
    meta = {
        "title": title,
        "url": row.get("Live URL", ""),
        "article_type": row.get("Article type", ""),
        "tags": [t.strip() for t in (row.get("Article Tags") or "").split(";") if t.strip()],
        "published": _parse_published(row.get("Published On", "")),
    }
    return [
        {"id": f"{row['slug']}-{i}", "text": f"{title}\n\n{body}", **meta}
        for i, body in enumerate(bodies)
    ]


def load_articles(csv_path) -> list[dict]:
    """Read the cleaned CSV, dropping the excluded promo articles."""
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["slug"] not in EXCLUDED_SLUGS]


def build_chunks(csv_path) -> list[dict]:
    return [chunk for row in load_articles(csv_path) for chunk in chunk_article(row)]


# --- embedding + artifact (network side; kept out of the pure seam above) ---

def embed(texts: list[str], batch_size: int = 100) -> "np.ndarray":
    import os
    import urllib.request

    import numpy as np
    from dotenv import load_dotenv

    load_dotenv(API_DIR.parent / ".env")
    key = os.environ["AI_GATEWAY_API_KEY"]
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        req = urllib.request.Request(
            "https://ai-gateway.vercel.sh/v1/embeddings",
            data=json.dumps({"model": "openai/text-embedding-3-small",
                             "input": batch}).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)["data"]
        vectors.extend(item["embedding"] for item in sorted(data, key=lambda d: d["index"]))
        print(f"embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.asarray(vectors, dtype=np.float32)


def main():
    import numpy as np

    chunks = build_chunks(CSV_PATH)
    print(f"{len(chunks)} chunks from {len(load_articles(CSV_PATH))} articles")
    vectors = embed([c["text"] for c in chunks])
    assert vectors.shape[0] == len(chunks)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.savez_compressed(ARTIFACT_DIR / "vectors.npz", vectors=vectors)
    print(f"wrote {ARTIFACT_DIR}/chunks.jsonl and vectors.npz {vectors.shape}")


if __name__ == "__main__":
    main()
