"""Seam-2 tests: CSV row in, chunk records out. Deterministic, no network."""

import tiktoken

from ingest import (
    EXCLUDED_SLUGS,
    MAX_TOKENS,
    MIN_CHUNK_CHARS,
    chunk_article,
    load_articles,
    merge_tiny,
    split_oversize,
    strip_html,
)

enc = tiktoken.get_encoding("cl100k_base")

LONG = "The Mustang's Coyote engine responds well to bolt-on modifications. " * 5  # >200 chars


def make_row(**overrides):
    row = {
        "Title": "Test Article",
        "slug": "test-article",
        "INTRO PARAGRAPH": f"<p>{LONG}</p>",
        "1 copy block": "",
        "2 copy block": "",
        "3 copy block": "",
        "4 copy block": "",
        "Article type": "Feature",
        "Article Tags": "coyote; gen-6-2015-2021-s550",
        "Published On": "Tue Sep 14 2021 11:32:05 GMT+0000 (Coordinated Universal Time)",
        "Live URL": "https://www.mustangdriver.com/feature-articles/test-article",
    }
    row.update(overrides)
    return row


def test_strip_html():
    html = '<p>Hello <strong>world</strong> &amp; more</p><h2>Heading</h2><ul><li>one</li><li>two</li></ul>'
    assert strip_html(html) == "Hello world & more\nHeading\none\ntwo"
    assert strip_html("") == ""
    assert strip_html("<p>&nbsp;</p>") == ""


def test_block_boundary_chunking():
    row = make_row(**{"1 copy block": f"<p>{LONG}A</p>", "2 copy block": f"<p>{LONG}B</p>"})
    chunks = chunk_article(row)
    assert len(chunks) == 3  # intro + 2 copy blocks, one chunk each
    assert all(c["text"].startswith("Test Article\n\n") for c in chunks)
    assert chunks[1]["text"].endswith("A")
    assert chunks[2]["text"].endswith("B")


def test_oversize_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(3000))  # ~6000 tokens
    pieces = split_oversize(text)
    assert len(pieces) > 1
    assert all(len(enc.encode(p)) <= MAX_TOKENS for p in pieces)
    for a, b in zip(pieces, pieces[1:]):  # adjacent pieces share overlapping text
        assert a[-40:] in b or b[:40] in a


def test_tiny_fragment_merge():
    assert merge_tiny([LONG, "tiny"]) == [LONG + "\n\ntiny"]
    assert merge_tiny(["tiny", LONG]) == ["tiny\n\n" + LONG]
    assert merge_tiny(["tiny"]) == ["tiny"]  # no neighbor: kept as-is
    for piece in merge_tiny([LONG + "A", "tiny", LONG + "B"]):
        assert len(piece) >= MIN_CHUNK_CHARS

    row = make_row(**{"1 copy block": "<p>tiny block</p>"})
    assert len(chunk_article(row)) == 1  # merged into the intro chunk


def test_promo_exclusion(tmp_path):
    assert len(EXCLUDED_SLUGS) == 21
    csv_path = tmp_path / "articles.csv"
    csv_path.write_text(
        "Title,slug\n"
        "Keep Me,some-article\n"
        "WIN This 1969 Mach 1 Mustang!,win-this-1969-mach-1-mustang\n"
    )
    rows = load_articles(csv_path)
    assert [r["slug"] for r in rows] == ["some-article"]


def test_metadata_presence():
    (chunk,) = chunk_article(make_row())
    assert chunk["id"] == "test-article-0"
    assert chunk["title"] == "Test Article"
    assert chunk["url"] == "https://www.mustangdriver.com/feature-articles/test-article"
    assert chunk["article_type"] == "Feature"
    assert chunk["tags"] == ["coyote", "gen-6-2015-2021-s550"]
    assert chunk["published"] == "2021-09-14"
