# Ask MustangDriver

An agentic RAG chatbot for [MustangDriver.com](https://www.mustangdriver.com) readers: ask any Mustang question in plain English, get an answer grounded in the site's article archive with clickable citations, tailored to *your* car — which the assistant remembers across visits without a login.

- **Live app:** https://ask-mustangdriver-web.vercel.app (works on phone and laptop browsers)
- **Live API:** https://ask-mustangdriver-api.vercel.app ([health](https://ask-mustangdriver-api.vercel.app/health))
- **Demo video (Loom):** *placeholder — link added after recording*
- **Full product spec:** [issue #1](https://github.com/theturpinator/ai-engineer-certification-challenge/issues/1)

This README is the written submission for the [certification challenge brief](requirements/README.md), structured one section per rubric task.

**Contents:** [Task 1](#task-1-problem-and-audience) · [Task 2](#task-2-proposed-solution) · [Task 3](#task-3-dealing-with-the-data) · [Task 4](#task-4-end-to-end-agentic-rag-prototype) · [Task 5](#task-5-evaluation-harness-and-baseline) · [Task 6](#task-6-improving-the-prototype) · [Task 7](#task-7-next-steps-for-demo-day) · [Quick start](#quick-start-local-development) · [Repo layout](#repo-layout)

---

## Task 1: Problem and audience

**Problem (one sentence):** A Mustang owner with a question about their car — mod guidance, install steps, model history, buying advice, recall status — has no fast, trustworthy way to get an answer that actually applies to their generation and trim.

**Why this is a problem.** The people with this problem are Mustang owners and enthusiasts — the readers of MustangDriver.com. They're trying to make real decisions about their cars: which mod to install next, how a part goes in, whether a used GT350 is worth the asking price, whether their model year has an open safety recall. Today they piece answers together from forum threads of unknown reliability, 20-minute YouTube videos that bury one useful detail, and MustangDriver.com's own archive of hundreds of expert articles — which can only be searched by keyword, so unless they guess the words the author used, the answer might as well not exist.

That isn't good enough for two reasons. First, none of it is tailored: Mustang advice is intensely generation-specific, and a Fox-body answer applied to an S550 is worse than no answer. Every source makes the owner do the filtering themselves, every single time. Second, the trustworthy source — the archive — is the hardest one to use, so owners bounce to Google and end up back in forum lore. (A secondary stakeholder, the site owner, wants readers engaging with the archive instead of bouncing — but this product is built for the reader, not the publisher.)

### How an owner answers a question today

```mermaid
flowchart TD
    Q["Mustang owner has a question:<br/>mod guidance, install steps, history, recalls"] --> F["Search Mustang forums"]
    F --> FP["Read multiple threads —<br/>unknown reliability, conflicting answers"]:::pain
    FP --> Y["Search YouTube"]
    Y --> YP["Scrub 20-minute videos<br/>for one detail"]:::pain
    YP --> A["Keyword-search the<br/>MustangDriver.com archive"]
    A --> AP["Exact-keyword match only —<br/>can't ask a question,<br/>can't combine articles"]:::pain
    AP --> X["Manually cross-check<br/>everything found"]
    X --> XP["Nothing is filtered to their<br/>generation/trim — a Fox-body answer<br/>can get applied to an S550"]:::pain
    XP --> R["Act on a best guess"]

    classDef pain fill:#ffdddd,stroke:#cc0000,color:#990000
```

Red nodes mark where the workflow is slow, repetitive, or error-prone: unverifiable sources, minutes of video for seconds of fact, brittle keyword search, and a tailoring step the owner must redo for every answer.

### Example evaluation questions

These are drawn from the committed golden set ([`evals/data/golden.jsonl`](evals/data/golden.jsonl)); categories map to the behaviors the app must get right.

| Question | Category — expected behavior |
|---|---|
| How much horsepower does the 2021 Mustang Mach 1 make? | `archive` — answer from the archive, with citation |
| Why was the 1971 Mustang the last of the big-block Mustangs? | `archive` — answer from the archive, with citation |
| Which assembly plants built the first classic Mustangs? | `archive` — answer from the archive, with citation |
| What is the towing capacity of the 2023 Ford F-150 Lightning? | `trap` — admit the archive doesn't cover it; don't invent |
| What was Ford's total Mustang production volume for the 2023 model year? | `trap` — admit the archive doesn't cover it; don't invent |
| Are there any open safety recalls on the 2020 Ford Mustang? | `recall` — route to the official NHTSA lookup |

## Task 2: Proposed solution

**Solution (one sentence):** "Ask MustangDriver" — a public web chat app that answers Mustang questions grounded in the MustangDriver.com article archive with clickable citations, transparently falls back to live web search or the official NHTSA recall API when the archive can't answer, and quietly remembers each user's car, mods, goals, and preferences across visits via a no-login browser identity, rendered on a "My Garage" page.

### Infrastructure

```mermaid
flowchart LR
    subgraph Browser["User's browser (phone or laptop)"]
        UI["Next.js chat + My Garage"]
    end
    UI -->|"SSE stream"| API["FastAPI on Vercel<br/>(Python Fluid Compute)"]
    API --> AGENT["LangGraph ReAct agent"]
    AGENT --> GW["Vercel AI Gateway"]
    GW --> S["Claude Sonnet 4.5<br/>(agent)"]
    GW --> H["Claude Haiku 4.5<br/>(summarizer)"]
    GW --> J["GPT-5-mini<br/>(RAGAS judge)"]
    GW --> E["OpenAI<br/>text-embedding-3-small"]
    AGENT --> QD["In-memory Qdrant<br/>(built from committed<br/>index artifact at cold start)"]
    AGENT --> PG["Neon Postgres<br/>(checkpointer +<br/>memory tables)"]
    AGENT --> TV["Tavily<br/>web search"]
    AGENT --> NH["NHTSA<br/>Recalls API"]
    API -.->|traces + experiments| LS["LangSmith"]
```

Why each component:

- **Agent LLM — Claude Sonnet 4.5:** chosen for tool-calling reliability and faithful citation behavior, the two things this agent does all day.
- **Summarizer LLM — Claude Haiku 4.5:** writes a rolling 2–3 sentence session summary every turn, so it needs to be cheap and fast, not brilliant.
- **Judge LLM — GPT-5-mini:** cheap at judge volume, and deliberately a different model family than the agent to avoid self-grading bias.
- **LLM gateway — Vercel AI Gateway:** all four models above are one config string and one API key, so swapping models is configuration rather than code.
- **Agent orchestration — LangGraph (`create_react_agent`):** a single tool-calling ReAct agent is the simplest shape that covers all five tools, and LangGraph's Postgres checkpointer gives durable conversation history for free.
- **Tools — `search_archive`, `web_search` (Tavily), `check_recalls` (NHTSA), `update_garage`, `update_instructions`:** Tavily covers the live web the archive can't; NHTSA is the free, keyless, *official* source for recall data; the two memory tools let conversation itself fill the user's profile.
- **Embedding model — OpenAI text-embedding-3-small:** strong price/performance baseline; text-embedding-3-large was A/B-tested in Task 6 and did not earn its cost.
- **Vector database — Qdrant, in-memory:** the corpus is only 993 chunks, so the serverless function rebuilds the collection from a committed artifact at cold start — no vector-DB service to run, pay for, or keep in sync.
- **Database — Neon Postgres (via Vercel Marketplace, pooled connection):** serverless-friendly Postgres for the LangGraph checkpointer and the three memory tables.
- **Monitoring — LangSmith:** every agent run is traced (reasoning, tool calls, memory writes are auditable), and eval runs land in LangSmith Experiments.
- **Evaluation framework — RAGAS + LangSmith Experiments:** RAG-specific metrics with an LLM judge, logged as comparable experiments (Task 5).
- **User interface — Next.js (React):** streaming chat and the My Garage page, responsive in phone and laptop browsers.
- **Deployment — Vercel:** two projects (static/SSR frontend + FastAPI as Python Fluid Compute functions) on one platform with one deploy story.

### Agent workflow

```mermaid
flowchart TD
    U["User message + browser UUID + session id"] --> P["Assemble per-turn system prompt:<br/>base persona + garage profile +<br/>standing instructions + recent session summaries"]
    P --> R{"Agent reasoning<br/>(Claude Sonnet 4.5,<br/>ReAct loop)"}
    R -->|"Mustang knowledge question"| SA["search_archive<br/>top-5 retrieval over Qdrant"]
    R -->|"live/current question or archive empty"| WS["web_search (Tavily)<br/>answer must disclose web sourcing"]
    R -->|"safety-recall question"| CR["check_recalls<br/>(NHTSA, by year/model)"]
    R -->|"user reveals car facts"| UG["update_garage<br/>(semantic memory)"]
    R -->|"user states a preference"| UP["update_instructions<br/>(procedural memory)"]
    SA --> R
    WS --> R
    CR --> R
    UG --> R
    UP --> R
    R -->|"answer ready"| O["Streamed tokens + archive citations<br/>rendered as clickable links"]
    O --> BG["Post-turn background task:<br/>Haiku 4.5 updates the rolling session summary<br/>(episodic memory)"]
```

Each turn starts by assembling a fresh system prompt from four parts: the base persona and routing policy, the user's garage profile (semantic memory), their standing instructions (procedural memory), and up to five recent past-session summaries (episodic memory) — so everything the assistant has ever learned about this user shapes the answer before reasoning even begins. The agent then runs a ReAct loop: for Mustang knowledge it calls `search_archive` first and grounds the answer in retrieved chunks, citing each source article as a markdown link; for inherently live questions (prices, news, events) or when the archive comes up empty it calls `web_search` and must open the answer with "According to a live web search" so the user can calibrate trust; recall questions route to `check_recalls` for official NHTSA campaign data. There is no human approval step — the transparency mechanisms (citations, web-search disclosure, admitting when no tool can answer) stand in for review.

Memory writes happen in the same loop: when the user mentions their car ("my 2019 GT with an intake"), the agent silently calls `update_garage`; when they state a standing preference ("keep answers short"), it calls `update_instructions`. The final answer streams token-by-token over SSE with a citations payload at the end, and after the stream closes, a background task has Claude Haiku 4.5 update a rolling 2–3 sentence summary of the session — which future sessions will see in their system prompt.

## Task 3: Dealing with the data

### Data source

The corpus is a cleaned CSV export of the MustangDriver.com article archive — **333 articles** (used with the site owner's permission), covering model history, builds, reviews, mod guides, and buying advice. **21 expired sweepstakes/giveaway promo articles were excluded** during data prep so readers are never told about contests that ended years ago; the exclusion rule (any article whose slug is a time-boxed sweepstakes/raffle/giveaway, identified by manual review) and the full 21-slug list are documented in [`api/EXCLUDED_ARTICLES.md`](api/EXCLUDED_ARTICLES.md), and ingestion re-checks the list by slug as a guard in case a promo re-enters a future export. The raw CSV stays out of the public repo; only the derived index artifact ([`api/index_artifact/`](api/index_artifact/) — chunk text + precomputed vectors) is committed, and the API builds its in-memory Qdrant collection from it at cold start.

### Chunking strategy

Chunks follow the CMS's **authored block boundaries** (intro paragraph + copy blocks) after stripping Webflow rich-text HTML. This is the default because those blocks are human-edited semantic units — an editor already decided where one idea ends and the next begins, which beats any fixed-size splitter on coherence. Two corrections handle the edge cases: blocks over ~1,000 tokens are recursively split with ~100 tokens of overlap (so no chunk blows past the embedding sweet spot), and fragments under 200 characters merge into a neighbor (so no chunk is too small to mean anything). Every chunk gets the **article title prepended** to its text — a chunk mid-article often never names its subject, and the title restores that context for both embedding and retrieval — plus metadata (title, live URL, article type, tags, published date) that powers citations. Result: **993 chunks across 333 articles** ([`api/ingest.py`](api/ingest.py)).

### External APIs and how they interact with RAG at usage time

Two external APIs complement retrieval, selected per-turn by the agent's tool policy:

- **Tavily (live web search):** the archive is a snapshot, so questions about current prices, market values, news, and events can't be answered from it. The agent calls Tavily only when the question is inherently live or archive retrieval comes up empty — RAG is always tried first for Mustang knowledge — and any web-built answer must be disclosed as web-sourced, keeping the trust boundary visible.
- **NHTSA Recalls API (free, keyless, official):** recall questions are safety-critical, so they bypass both the archive and the web in favor of authoritative government data queried by year/make/model. The RAG layer is deliberately *not* used for recalls; the eval golden set includes recall-routing questions to verify this.

## Task 4: End-to-end agentic RAG prototype

The full vertical slice is built and **deployed publicly**:

- **Frontend:** https://ask-mustangdriver-web.vercel.app — Next.js streaming chat with markdown rendering, inline archive citations as clickable links, and a **My Garage** page showing everything the assistant has learned (car, mods, wishlist, goals, standing instructions, recent session summaries).
- **Backend:** https://ask-mustangdriver-api.vercel.app — FastAPI on Vercel Python Fluid Compute. `POST /chat` streams SSE events (tool notifications, tokens, then a citations payload); `GET /garage/{user_id}` returns the profile, instructions, and summaries; `GET /health` for checks. Full contract in [`api/README.md`](api/README.md).
- **Agent:** a single LangGraph ReAct agent (Claude Sonnet 4.5 via the Vercel AI Gateway) with the five tools described in Task 2, retrieval over in-memory Qdrant, and LangSmith tracing on every run.

**Three-type memory model** (all keyed by a UUID minted into `localStorage` on first visit — no accounts, no login):

| Type | What | Where |
|---|---|---|
| **Semantic** | Garage profile: car year/trim/generation, installed mods, wishlist, goals — written by the `update_garage` tool as the user chats | `garage` table (one JSONB row per user, partial updates merge) |
| **Episodic** | Live thread history via LangGraph's Postgres checkpointer, plus a rolling 2–3 sentence per-session summary written by Claude Haiku 4.5 after every turn; recent past-session summaries are injected into the system prompt | Checkpointer tables + `summaries` table |
| **Procedural** | Standing preferences ("keep answers short") written by the `update_instructions` tool, appended to the system prompt every turn | `instructions` table |

Identity is browser-scoped: the UUID doubles as the LangGraph thread id, so conversation and profile survive page refreshes and return visits with zero friction; a per-tab session id (in `sessionStorage`) separates episodic summaries by visit. Local Postgres runs in Docker; production is Neon (pooled connection string). Tests cover the FastAPI HTTP boundary (chat with citations, memory round-trips, tool-routing smoke suite) and the ingestion pipeline ([`api/tests/`](api/tests/)).

## Task 5: Evaluation harness and baseline

### Test data

- **Synthetic set — 51 samples** ([`evals/data/testset.jsonl`](evals/data/testset.jsonl)): generated from the corpus with RAGAS's knowledge-graph testset generation ([`evals/generate_testset.py`](evals/generate_testset.py)).
- **Golden set — 10 hand-written Q/A pairs** ([`evals/data/golden.jsonl`](evals/data/golden.jsonl)): 6 archive-answerable, 2 out-of-archive **traps** (correct behavior = admit the archive doesn't cover it), 2 **recall-routing** questions (correct behavior = defer to the official NHTSA lookup). Reference answers are currently marked `"status": "stub"` — the author's best-effort answers, to be refined into verified ground truth.

### Harness

[`evals/run_eval.py`](evals/run_eval.py) scores the **RAG pipeline in isolation** — retriever + fixed-prompt generation with Claude Sonnet 4.5 ([`evals/rag_pipeline.py`](evals/rag_pipeline.py)), separated from the production agent so A/B comparisons measure only the changed variable. Metrics: **faithfulness, answer relevancy, context precision, context recall** on the synthetic set, plus **answer correctness** on the golden set. The judge is **GPT-5-mini** (different family than the agent, to dodge self-grading), all via the Vercel AI Gateway. Every run logs to **LangSmith Experiments** against the `ask-mustangdriver-synthetic` / `ask-mustangdriver-golden` datasets. Details and gateway gotchas: [`evals/README.md`](evals/README.md).

### Baseline results

| Variant | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness |
|---|---:|---:|---:|---:|---:|
| baseline (dense top-5, 3-small) | 0.9187 | 0.6358 | 0.7564 | 0.6638 | 0.6473 |

### Conclusions from the baseline

- **Generation is not the problem.** Faithfulness of 0.92 means that when relevant context is retrieved, the model grounds its answer in it rather than inventing — the fixed prompt's "answer only from context, admit gaps" policy holds.
- **Retrieval is the bottleneck.** Context recall of 0.66 says dense retrieval misses roughly a third of the context needed for a complete answer, and precision of 0.76 says the top-5 carries noise. Both point at the retriever, not the generator — which is exactly what Task 6 targets.
- **Answer relevancy (0.64) is deflated partly by design:** trap-style questions the pipeline correctly refuses to answer ("the archive doesn't cover this") score low on relevancy even though refusal is the desired behavior.
- **Answer correctness (0.65) is a soft signal for now** — it's measured against 10 stub reference answers, so it can rank variants but shouldn't be over-read in absolute terms.

## Task 6: Improving the prototype

### Advanced retrieval: hybrid BM25 + dense with reciprocal rank fusion

Mustang jargon is **exact-match heavy** — chassis codes (S550, SN95), model names (GT350, Cobra Jet, Dark Horse), engine designations ("5.0 Coyote") — and dense embeddings blur these near-identical alphanumeric codes together. Hybrid retrieval keeps dense's semantic reach while letting BM25 nail the exact tokens: BM25 (`rank_bm25`, with a tokenizer that keeps "S550" and "5.0" whole) and dense rankings are fused with reciprocal rank fusion (k=60) over each ranker's top-50.

### Second change: embedding model A/B

The other component change swaps `text-embedding-3-small` → `text-embedding-3-large` (index fully re-embedded). Both changes run through the identical harness in one experiments notebook ([`evals/experiments.ipynb`](evals/experiments.ipynb)), producing a 2×2 comparison.

### Results (full table, from [`evals/results/comparison.md`](evals/results/comparison.md))

| Variant | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness |
|---|---:|---:|---:|---:|---:|
| baseline (dense, 3-small) | 0.9187 | 0.6358 | 0.7564 | 0.6638 | 0.6473 |
| **hybrid RRF (3-small)** | **0.9301** | **0.6560** | 0.7907 | **0.7426** | 0.6422 |
| dense (3-large) | 0.9158 | 0.5897 | 0.7645 | 0.6427 | 0.6398 |
| hybrid RRF (3-large) | 0.9278 | 0.6508 | **0.7955** | 0.6893 | 0.6316 |

Per-variant aggregates are in [`evals/results/`](evals/results/); all 8 runs (4 variants × synthetic/golden) are logged as LangSmith Experiments prefixed `baseline-dense-`, `hybrid-rrf-`, `dense-3large-`, `hybrid-3large-`.

### Conclusion: what improved and why

**Hybrid RRF on 3-small embeddings is the winner, and the improvement is exactly where the baseline said the problem was — retrieval quality.** Context recall jumped **+0.079** (0.664 → 0.743) and context precision **+0.034** (0.756 → 0.791): BM25's exact-token matching recovers the jargon-keyed chunks that dense embeddings blur, and RRF folds them in without sacrificing dense's semantic matches. Better context propagates downstream — faithfulness (+0.011) and answer relevancy (+0.020) both tick up, because the generator has more of the right material to work with.

**The bigger embedding model was not worth it.** Dense 3-large is a wash-to-negative versus 3-small (recall −0.021, relevancy −0.046) at higher cost, and pairing it with hybrid still loses to hybrid + 3-small on recall (0.689 vs 0.743). On this corpus's specialized vocabulary, lexical matching adds real signal; more embedding dimensions don't.

**Answer correctness is flat across all four variants** (0.63–0.65) — expected, since it's judged against 10 stub-referenced golden answers and most golden questions are answerable from multiple retrieval configurations; it will become a sharper discriminator once the author's verified answers land.

**Decision: ship hybrid RRF + text-embedding-3-small.** Honest caveat: the hybrid retriever currently lives in the eval pipeline ([`evals/rag_pipeline.py`](evals/rag_pipeline.py)); promoting it into the production agent's `search_archive` tool is a Demo Day item (Task 7).

## Task 7: Next steps for Demo Day

### Keep

The whole tracer-bullet spine earned its place and stays: the two-app Vercel monorepo, the single LangGraph ReAct agent (no router or multi-agent graph proved necessary), the five-tool design, the three-type memory model with no-login browser identity, the committed-index-artifact ingestion (zero vector-DB ops), the Vercel AI Gateway model routing, LangSmith tracing, and the isolated RAGAS + LangSmith eval harness that makes every future change measurable the same way.

### Change or add

Near-term engineering improvements, straight from this write-up's honest caveats:

1. **Promote the hybrid RRF retriever into the production agent** — it won the eval but currently only exists in the eval pipeline; `search_archive` still runs dense-only.
2. **Replace the golden-set stubs with the author's verified answers** so answer correctness becomes a trustworthy discriminator, and grow the set beyond 10 samples.
3. **Agent-level evaluation** — the harness scores the RAG pipeline in isolation; adding tool-routing and memory-write evals over the full agent would cover the behaviors the golden traps only smoke-test today.

Product items deliberately deferred out of the certification scope (see [issue #1](https://github.com/theturpinator/ai-engineer-certification-challenge/issues/1)) that form the Demo Day roadmap:

- **Garage as a destination — largely SHIPPED post-spec** as the increment documented in [issue #13](https://github.com/theturpinator/ai-engineer-certification-challenge/issues/13) (2026-07-11): the garage now holds multiple cars (chat-driven car targeting, legacy profiles migrate), each with Need-for-Speed-style 0–100 stat bars plus real stock figures (LLM-generated once per car and cached), an AI-generated car portrait (gpt-image-1 via the gateway, cached server-side, regenerated when identity/color changes), a car switcher, and inline editing (PATCH/DELETE endpoints) — chat remains the primary input; forms are an override for correcting what the assistant learned. Still open from this item: stat-boost estimates for proposed mods.
- **Partner-vendor product sourcing** and vendor-restricted search, so mod suggestions can link to real parts — the publisher-side monetization story.
- **Live Webflow CMS sync** replacing the CSV snapshot, so new articles enter the index as they're published.
- **Accounts / cross-device identity** — localStorage UUID is zero-friction but device-bound; an optional lightweight login would carry the garage between phone and laptop.
- **Production hardening:** rate limiting and abuse protection (the API is currently open), plus CI/CD for tests and evals on every change.
- **Native mobile apps remain out** — the browser experience already works on the garage floor, so this stays cut until users say otherwise.

## Quick start (local development)

Prereqs: Python 3.12+ with [`uv`](https://docs.astral.sh/uv/), Node 20+, Docker.

```sh
cp .env.example .env         # fill in AI_GATEWAY_API_KEY, TAVILY_API_KEY, LANGSMITH_*
docker compose up -d         # Postgres on host port 5433

# API (http://localhost:8000)
cd api
uv venv && uv pip install -r requirements.txt
uv run uvicorn app:app --port 8000

# Web (http://localhost:3000) — in another terminal
cd web
npm install
npm run dev
```

Local `DATABASE_URL` is `postgresql://postgres:postgres@localhost:5433/mustang`; production uses the Neon pooled connection string. API tests: `cd api && uv run pytest`. Eval harness setup and runs: [`evals/README.md`](evals/README.md).

## Repo layout

```
api/            FastAPI backend: agent, tools, memory, ingestion, tests
  index_artifact/   committed chunks + vectors (the corpus ships pre-embedded)
  EXCLUDED_ARTICLES.md   promo-exclusion rule + full list of 21
evals/          RAGAS harness, datasets, experiments notebook, results
  results/          per-variant aggregates + comparison.md
web/            Next.js frontend: streaming chat + My Garage
requirements/   the certification challenge brief this README answers
docker-compose.yml   local Postgres
.env.example    required environment variables
```

**Monitoring:** all agent runs and eval experiments are in LangSmith under the project `ask-mustangdriver` (traces show reasoning, tool calls, and memory writes; Experiments hold the 8 eval runs behind the Task 6 table).

**Demo video:** *Loom link placeholder — the author records the 10-minute demo separately.*
