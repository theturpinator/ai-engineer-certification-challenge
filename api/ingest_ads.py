"""Ads ingestion: advertiser CSV export -> committed product catalog artifact.

Run from api/ with the venv set up (see README.md):

    uv run python -m ingest_ads

Reads ../data/ads.csv (the Webflow advertiser export, gitignored). Every row
enters the pipeline regardless of the export's active flag (issue #50, a
demo-scope decision). Near-duplicate rows (repeat giveaway campaigns, repeat
event years, banner variants) are deduplicated before analysis: key = the
canonical website's domain, fallback = normalized advertiser name; the
newest row by the export's updated-on column wins and keeps the union of
the merged rows' creatives. Each advertiser's canonical website is derived
by a pure fallback chain (website column -> google-ad column with HTML
stripped -> origin of any banner click-through link), then confirmed or
corrected by the vision analysis. Each advertiser's creative
images are vision-analyzed by the agent model (Sonnet via the AI Gateway) to
classify the advertiser (product vendor / service / event / charity /
giveaway / placeholder) and extract concrete products with categories,
keywords, and aliases. Recommendation eligibility = product-or-
service vendor; the AdSense placeholder, charities, and giveaways are
ingested but never recommendable. Every vendor/service advertiser also gains
one advertiser-level entry (kind = advertiser): recommendable, Sponsored,
linking to the canonical website, embedded under the advertiser's name,
description, and the union of its products' categories — how the agent
answers "where should I buy X" with a sponsor's website. Advertiser entries
are never Upgrade Shop-eligible. Products additionally carry a `specific`
flag (a concrete installable product vs a broad product line or service):
only specific products appear in the garage Upgrade Shop; broad lines and
services stay chat-recommendable only. Generic mod categories (supercharger,
exhaust, ...) join the catalog unbranded. Every entry carries per-generation
deltas for the nine garage stats — five performance plus four ownership
(style, comfort, safety, reliability) — generated once here with
conservative calibration (performance deltas never invented for
non-performance goods, which earn ownership deltas instead), and every entry is
embedded, so the runtime index needs no network at startup. Creative images
stay hotlinked CDN URLs; click-through links keep their existing UTM
parameters. Writes:

    ads_artifact/catalog.jsonl   one product/category entry per line
    ads_artifact/vectors.npz     float32 vectors, row i matches catalog line i

Re-running against a fresh export is the roster-refresh path — no code change.
"""

import base64
import csv
import json
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

API_DIR = Path(__file__).parent
CSV_PATH = API_DIR.parent / "data" / "ads.csv"
ARTIFACT_DIR = API_DIR / "ads_artifact"

MODEL = "anthropic/claude-sonnet-4.5"  # the agent model

PERFORMANCE_STATS = ("power", "acceleration", "top_speed", "handling", "braking")
# Ownership stats (issue #27): what non-performance products honestly move —
# dash cam -> safety, seat covers -> comfort, cooling/resto -> reliability,
# paint/coatings/wheels/exhaust-sound -> style.
OWNERSHIP_STATS = ("style", "comfort", "safety", "reliability")
STATS = PERFORMANCE_STATS + OWNERSHIP_STATS
# Must match app._derive_generation's names — deltas are keyed by these.
GENERATION_NAMES = ("First generation", "Mustang II", "Fox-body", "SN95",
                    "S197", "S550", "S650")

ELIGIBLE_CLASSIFICATIONS = frozenset({"product vendor", "service"})

# Creative image columns paired with their click-through link columns,
# in card-display preference order (square renders best in a card).
CREATIVE_COLUMNS = (
    ("300x250 square image", "300x250 square ad link"),
    ("Client banner Ad", "Banner Ad link"),
    ("Small Banner Image", "Small banner link"),
)

# Unbranded default mod categories: build planning isn't limited to whoever
# advertises. No advertiser, no creative, no link — deltas and aliases only.
GENERIC_MODS = (
    ("Supercharger", ["forced induction"],
     ["blower", "whipple", "procharger", "roush supercharger", "kenne bell"]),
    ("Turbocharger", ["forced induction"],
     ["turbo", "twin turbo", "turbo kit"]),
    ("Cold air intake", ["intake"], ["cai", "intake", "air intake"]),
    ("Cat-back exhaust", ["exhaust"],
     ["exhaust", "catback", "axle-back exhaust", "flowmaster", "borla", "muffler"]),
    ("Long tube headers", ["exhaust"], ["headers", "shorty headers"]),
    ("Lowering springs / coilovers", ["suspension"],
     ["lowering springs", "coilovers", "springs", "sway bars", "suspension"]),
    ("Big brake kit", ["brakes"], ["brakes", "brembo brakes", "brake kit"]),
    ("Performance tune", ["electronics"],
     ["tune", "ecu tune", "dyno tune", "flash tune", "tuner"]),
    ("Wheels and tires", ["wheels", "tires"],
     ["wheels", "tires", "rims", "wheel and tire package"]),
    ("Performance transmission", ["transmission"],
     ["transmission", "transmission swap", "manual swap", "gearbox"]),
    ("Rear gears", ["drivetrain"],
     ["gears", "differential", "3.73 gears", "4.10 gears", "rear end"]),
    ("Nitrous kit", ["engine"], ["nitrous", "nos"]),
    ("Camshafts", ["engine"], ["cams", "camshaft"]),
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def zero_deltas() -> dict:
    return {g: {s: 0 for s in STATS} for g in GENERATION_NAMES}


def load_advertisers(csv_path) -> list[dict]:
    """Every row in the export — the active flag no longer gates ingestion
    (issue #50); the classification gate still rules recommendability."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


_URL_RE = re.compile(r"https?://[^\s\"'<>]+|www\.[^\s\"'<>]+")


def _first_url(text: str) -> str:
    """The first URL in free text, HTML tags stripped; '' when none."""
    m = _URL_RE.search(re.sub(r"<[^>]+>", " ", text or ""))
    if not m:
        return ""
    url = m.group().rstrip(".,;)")
    return url if url.startswith("http") else f"https://{url}"


def canonical_website(row: dict) -> str:
    """Mechanical canonical-website chain (issue #50): the website column,
    else the google-ad column (HTML stripped), else the origin (scheme +
    host) of any creative click-through link. The vision analysis later
    confirms/corrects the result (campaign microsites, CDN artifacts)."""
    for col in ("Website Link", "google ad link"):
        url = _first_url(row.get(col) or "")
        if url:
            return url
    for _img_col, link_col in CREATIVE_COLUMNS:
        url = _first_url(row.get(link_col) or "")
        if url:
            p = urlparse(url)
            return f"{p.scheme}://{p.netloc}"
    return ""


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_updated(raw: str) -> datetime:
    """Webflow's 'Tue Apr 05 2022 19:07:37 GMT+0000 (...)'; min when blank."""
    try:
        return datetime.strptime(" ".join(raw.split()[:5]), "%a %b %d %Y %H:%M:%S")
    except ValueError:
        return datetime.min


# ponytail: link shorteners aren't advertiser identities — extend if the
# roster ever gains more of them
_SHORTENER_DOMAINS = frozenset({"amzn.to"})


def dedupe_records(records: list[dict]) -> list[dict]:
    """Near-duplicate advertisers collapse (issue #50): same canonical-website
    domain (else normalized name) = same advertiser. The newest row wins and
    keeps the union of the merged rows' creatives, so no product is lost."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        dom = _domain(rec["website"])
        key = dom if dom and dom not in _SHORTENER_DOMAINS else _norm_name(rec["name"])
        groups.setdefault(key, []).append(rec)
    out = []
    for group in groups.values():
        group.sort(key=lambda r: r["updated"], reverse=True)
        survivor = group[0]
        seen = {c["image"] for c in survivor["creatives"]}
        for other in group[1:]:
            for c in other["creatives"]:
                if c["image"] not in seen:
                    seen.add(c["image"])
                    survivor["creatives"].append(c)
        out.append(survivor)
    return out


def advertiser_record(row: dict) -> dict:
    """Name, slug, canonical website, updated-on (the dedup tiebreaker), and
    creatives (image + its click-through link, UTM untouched), deduped, in
    card-display preference order (square first)."""
    website = canonical_website(row)
    creatives, seen = [], set()
    for img_col, link_col in CREATIVE_COLUMNS:
        img = (row.get(img_col) or "").strip()
        if img and img not in seen:
            seen.add(img)
            creatives.append(
                {"image": img, "link": (row.get(link_col) or "").strip() or website}
            )
    return {
        "name": row["Advertisers Name"].strip(),
        "slug": row["Slug"].strip(),
        "website": website,
        "updated": _parse_updated(row.get("Updated On") or ""),
        "creatives": creatives,
    }


def catalog_entries(record: dict, analysis: dict) -> list[dict]:
    """Sponsor catalog entries for one advertiser given its vision analysis.

    Vendor/service advertisers yield one recommendable entry per extracted
    product plus one advertiser-level entry (kind = advertiser, issue #50) —
    Sponsored, linking to the canonical website, embedded under the union of
    its products' categories, never Upgrade Shop-eligible. Anything else
    (placeholder, charity, giveaway, event) stays a single non-recommendable
    entry — ingested, never recommended. Recommendable product entries leave
    deltas None for the impure delta step to fill."""
    cls = analysis.get("classification", "")
    creative = (record["creatives"] or [{"image": None, "link": record["website"] or None}])[0]
    base = {
        "advertiser": record["name"],
        "advertiser_slug": record["slug"],
        "sponsored": True,
        "classification": cls,
        "image": creative["image"],
        "link": creative["link"] or record["website"] or None,
    }
    if cls not in ELIGIBLE_CLASSIFICATIONS:
        return [{
            **base,
            "id": record["slug"],
            "name": record["name"],
            "recommendable": False,
            "specific": False,
            "description": analysis.get("description", ""),
            "categories": [],
            "keywords": [],
            "aliases": [],
            "deltas": zero_deltas(),
        }]
    products = [{
        **base,
        "id": f"{record['slug']}-{slugify(p['name'])}",
        "name": p["name"],
        "recommendable": True,
        # Upgrade Shop-eligible only if it's a concrete installable product;
        # services and broad product lines stay chat-recommendable only.
        "specific": bool(p.get("specific")) and cls == "product vendor",
        "description": p.get("description", ""),
        "categories": p.get("categories", []),
        "keywords": p.get("keywords", []),
        "aliases": p.get("aliases", []),
        "deltas": None,
    } for p in analysis.get("products", [])]
    advertiser = {
        **base,
        "id": record["slug"],
        "kind": "advertiser",
        "name": record["name"],
        "recommendable": True,
        "specific": False,  # a storefront, not an installable product
        "description": analysis.get("description", ""),
        "categories": sorted({c for p in products for c in p["categories"]}),
        "keywords": [],
        "aliases": [],
        "link": record["website"] or base["link"],
        "deltas": zero_deltas(),  # an advertiser isn't an upgrade
    }
    return products + [advertiser]


def generic_entries() -> list[dict]:
    """The unbranded default mod categories, deltas left for the impure step."""
    return [{
        "id": "generic-" + slugify(name),
        "name": name,
        "advertiser": None,
        "advertiser_slug": None,
        "sponsored": False,
        "classification": "generic",
        "recommendable": False,
        "specific": True,
        "description": f"Generic {name.lower()} upgrade for any Mustang.",
        "categories": categories,
        "keywords": aliases,
        "aliases": aliases,
        "image": None,
        "link": None,
        "deltas": None,
    } for name, categories, aliases in GENERIC_MODS]


def normalize_deltas(raw: dict) -> dict:
    """Coerce an LLM deltas reply to the committed shape: every generation,
    every stat, int values; anything missing or malformed becomes 0."""
    out = {}
    for g in GENERATION_NAMES:
        gen = raw.get(g) or {}
        row = {}
        for s in STATS:
            try:
                row[s] = int(gen.get(s, 0))
            except (TypeError, ValueError):
                row[s] = 0
        out[g] = row
    return out


def embed_text(entry: dict) -> str:
    """The text a catalog entry is embedded (and BM25-indexed) under."""
    return "\n".join(filter(None, [
        entry["name"],
        entry.get("advertiser") or "",
        entry.get("description", ""),
        "categories: " + ", ".join(entry.get("categories", [])),
        "keywords: " + ", ".join(entry.get("keywords", [])),
    ]))


# --- vision / LLM / embedding side (network; kept out of the pure seam) ---

ANALYSIS_PROMPT = """You are cataloging advertisers for MustangDriver.com, a \
Ford Mustang enthusiast site. Analyze this advertiser and their ad creative \
image(s).

Advertiser name: {name}
Website: {website}
Ad click-through link: {link}

Classify the advertiser and extract what they sell. Reply with ONLY this JSON:
{{"classification": "product vendor" | "service" | "event" | "charity" | \
"giveaway" | "placeholder",
 "website": "<the advertiser's canonical homepage URL — correct campaign \
microsites, CDN artifacts, and link-shortener URLs to the real homepage when \
you know it; '' if unknown>",
 "description": "<one line: who this advertiser is>",
 "products": [{{"name": "<specific product or product line, not just the brand>",
   "specific": <true only for a concrete installable product an owner could \
point to on or in the car (a named part, kit, or device); false for broad \
product lines, whole catalogs, and services>,
   "description": "<one line: what it is and why a Mustang owner would want it>",
   "categories": ["<one or two of: transmission, drivetrain, engine, \
forced induction, intake, exhaust, suspension, brakes, wheels, tires, \
electronics, exterior, interior, paint, restoration parts, apparel, travel, \
other>"],
   "keywords": ["<search terms a shopper would use to find it>"],
   "aliases": ["<short names an owner might use for it in a mod list>"]}}]}}

Rules:
- "placeholder" is for ad-network placeholders (e.g. Google AdSense); \
"giveaway" for sweepstakes/raffle campaigns; "charity" for charitable \
organizations.
- Extract 1-3 concrete products or product lines actually shown in the \
creatives or clearly sold by this advertiser.
- For any classification other than "product vendor" or "service", \
"products" must be []."""

DELTAS_PROMPT = """You are calibrating an arcade-style Mustang build game. \
Nine ratings, each a 0-100 score calibrated across the whole Mustang range:
- Performance: power, acceleration, top_speed, handling, braking (a stock \
2015+ GT is roughly 75-85 power, a 2020 Shelby GT500 is 95-100).
- Ownership: style, comfort, safety, reliability (a 1965 coupe rates low on \
safety and comfort, an S650 high).

For the upgrade below, give the rating DELTA (integer change) it makes to a \
Ford Mustang of each generation. Be conservative and realistic.

Upgrade: {name}
Description: {description}
Categories: {categories}

Rules:
- A typical bolt-on adds +1 to +5 on the performance stats it affects; \
forced induction adds +10 to +20 power.
- NEVER invent performance value: a product that does not change how the \
car drives (cosmetics, electronics, apparel, services, travel, restoration/\
replacement parts) gets zero on all five performance stats.
- Ownership deltas are where such products earn their keep, when genuine: \
paint, coatings, lighting, wheels, and exhaust sound/character move style; \
seats and interior move comfort; cameras, brakes, tires, and lighting move \
safety; cooling, restoration/replacement parts, and drivetrain refreshes \
move reliability. Typical honest ownership delta: +2 to +8.
- 0 for any stat the upgrade doesn't clearly affect; negative only when it \
genuinely hurts a stat. Pure apparel/travel/event items get all zeros \
everywhere.

Reply with ONLY this JSON (integer values, all seven generations, all nine \
stats per generation):
{{"First generation": {{"power": 0, "acceleration": 0, "top_speed": 0, \
"handling": 0, "braking": 0, "style": 0, "comfort": 0, "safety": 0, \
"reliability": 0}}, "Mustang II": {{...}}, "Fox-body": {{...}}, \
"SN95": {{...}}, "S197": {{...}}, "S550": {{...}}, "S650": {{...}}}}"""


def _gateway(path: str, payload: dict) -> dict:
    import os

    import httpx
    from dotenv import load_dotenv

    load_dotenv(API_DIR.parent / ".env")
    resp = httpx.post(
        f"https://ai-gateway.vercel.sh/v1{path}",
        headers={"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def _chat_json(content) -> dict:
    """One gateway chat call; the first JSON object in the reply."""
    resp = _gateway("/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
    })
    text = resp["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model reply: {text[:200]}")
    return json.loads(m.group())


def analyze_advertiser(record: dict) -> dict:
    """Vision-analyze an advertiser's creatives: classification + products."""
    import httpx

    content = [{"type": "text", "text": ANALYSIS_PROMPT.format(
        name=record["name"], website=record["website"] or "unknown",
        link=(record["creatives"][0]["link"] if record["creatives"] else "") or "unknown",
    )}]
    for creative in record["creatives"][:3]:
        img = httpx.get(creative["image"], timeout=60, follow_redirects=True)
        img.raise_for_status()
        media = mimetypes.guess_type(creative["image"].split("?")[0])[0] or "image/jpeg"
        b64 = base64.b64encode(img.content).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{b64}"}})
    return _chat_json(content)


def generate_deltas(entry: dict) -> dict:
    """Per-generation stat deltas for one catalog entry, normalized."""
    raw = _chat_json(DELTAS_PROMPT.format(
        name=entry["name"], description=entry["description"],
        categories=", ".join(entry["categories"]),
    ))
    return normalize_deltas(raw)


def main():
    import numpy as np

    from ingest import embed  # same gateway embedding path as the article index

    rows = load_advertisers(CSV_PATH)
    records = dedupe_records([advertiser_record(r) for r in rows])
    print(f"{len(rows)} roster rows -> {len(records)} advertisers after dedup")
    entries = []
    for record in records:
        analysis = analyze_advertiser(record)
        # the vision call confirms/corrects the mechanical website chain
        record["website"] = _first_url(analysis.get("website", "")) or record["website"]
        made = catalog_entries(record, analysis)
        print(f"  {record['name']}: {analysis.get('classification')} -> "
              f"{len(made)} entries ({'recommendable' if made and made[0]['recommendable'] else 'not recommendable'})")
        entries.extend(made)
    entries.extend(generic_entries())
    for entry in entries:
        if entry["deltas"] is None:
            entry["deltas"] = generate_deltas(entry)
            moved = {g: d for g, d in entry["deltas"].items() if any(d.values())}
            print(f"  deltas {entry['id']}: {moved if moved else 'all zero'}")
    vectors = embed([embed_text(e) for e in entries])
    assert vectors.shape[0] == len(entries)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_DIR / "catalog.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    np.savez_compressed(ARTIFACT_DIR / "vectors.npz", vectors=vectors)
    recommendable = sum(e["recommendable"] for e in entries)
    print(f"wrote {ARTIFACT_DIR}/catalog.jsonl ({len(entries)} entries, "
          f"{recommendable} recommendable) and vectors.npz {vectors.shape}")


if __name__ == "__main__":
    main()
