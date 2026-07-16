# Ask MustangDriver

An agentic RAG chatbot for [MustangDriver.com](https://www.mustangdriver.com) readers: ask any Mustang question in plain English, get an answer grounded in the site's article archive with clickable citations and advertiser product recommendations. The agent remembers key details about the customers vehicle, such as model, color, installed upgrades, and a wishlist of future upgrades.

- **Live app:** https://ask-mustangdriver-web.vercel.app (works on phone and laptop browsers)
- **Live API:** https://ask-mustangdriver-api.vercel.app ([health](https://ask-mustangdriver-api.vercel.app/health))
- **Demo video (Loom):** *placeholder — link added after recording*
- **Full product spec:** [issue #1](https://github.com/theturpinator/ai-engineer-certification-challenge/issues/1)

This README is the written submission for the certification challenge brief.

**Contents:** [Task 1](#task-1-problem-and-audience) · [Task 2](#task-2-proposed-solution) · [Task 3](#task-3-dealing-with-the-data) · [Task 4](#task-4-end-to-end-agentic-rag-prototype) · [Task 5](#task-5-evaluation-harness-and-baseline) · [Task 6](#task-6-improving-the-prototype) · [Task 7](#task-7-next-steps-for-demo-day) · [Quick start](#quick-start-local-development) · [Repo layout](#repo-layout)

---

## Task 1: Problem and audience

**Problem (one sentence):** MustangDriver has poor customer engagement, and lacks customer profiles for proper product, event, and article recommendations.

**Why this is a problem.** The Mustang fanbase is huge - in 2018, Ford sold its 10 millionth Mustang. The brand continues to grow, and its fans are die hard. There are countless Mustang-focused events, communities, and media outlets. MustangDriver is a digital magazine that covers all things Mustang, including event coverage, upgrade tutorials, and car reviews. Today, they have approximately 28k subscribers, most of which they have no information about. They claim to teach customers to "Buy Right" (meaning, buy the right car for them and buy the right upgrades for their vehicle), but they lack any information about the customer's vehicle, goals, or interests.

Without a proper customer profile, MustangDriver cannot provide a tailored experience for their users. They want to personalize the customer experience and direct them to articles and products to boost engagement and customer satisfaction. Today, personalization is only acheivable through their naive search feature, which directs them to entire articles based on keyword search. This isn't good enough because the search results don't take into consideration an individual users preferences and lacks an interactive and directed form of information delivery.

### How a customer finds information today

```mermaid
flowchart TD
    Q["MustangDriver subscriber<br/>enters site"] --> S["Keyword search on<br/>MustangDriver.com"]
    S --> SP["Exact-keyword match</br>returns articles"]:::pain
    SP --> RD["Read entire articles to find<br/>single relevant detail"]:::pain
    RD --> G{"Question Answered?"}
    G -->|"Yes"| R["Reaches Conclusion"]
    G -->|"No — information gap"| L["Leave MustangDriver.com for<br/>forums, YouTube, Google"]:::pain
    L --> X["Cross-check conflicting,<br/>unverified answers off-site"]:::pain
    X --> R
    R --> N["MustangDriver learns</br>nothing about user;</br>potential churn to competitor"]:::pain

    classDef pain fill:#ffdddd,stroke:#cc0000,color:#990000
```

### Example evaluation questions

These are drawn from the committed golden set ([`evals/data/golden.jsonl`](evals/data/golden.jsonl)); categories map to the behaviors the app must get right.

| Question | Category — expected behavior |
|---|---|
| How much horsepower does the 2021 Mustang Mach 1 make? | `archive` — answer from the archive, with citation |
| Why was the 1971 Mustang the last of the big-block Mustangs? | `archive` — answer from the archive, with citation |
| Which assembly plants built the first classic Mustangs? | `archive` — answer from the archive, with citation |
| What is the towing capacity of the 2023 Ford F-150 Lightning? | `guardrail` — should mention that it only covers Mustangs |
| What was Ford's total Mustang production volume for the 2023 model year? | `fallback to tool` — user web search to fill gap in archive |
| Are there any open safety recalls on the 2020 Ford Mustang? | `recall` — route to the official NHTSA lookup |

## Task 2: Proposed solution

**Solution (one sentence):** "Ask MustangDriver" - a chat interface that keeps track of a subscriber's car(s) in a virtual garage, while additionally tracking their interests and desired car upgrades.

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

- **Agent LLM — Claude Sonnet 4.5:** chosen for tool-calling reliability, faithful citation behavior, and superior instruction following capabilities.
- **Summarizer LLM — Claude Haiku 4.5:** Chosen for writing session summaries (episodic memory) to save on token costs (Sonnet is a little too much horsepower for this type of work).
- **Judge LLM — GPT-5-mini:** an affordable model and deliberately a different model family than the agent to avoid self-grading bias.
- **LLM gateway — Vercel AI Gateway:** all 3 models above are one config string and one API key, so swapping models is configuration rather than code.
- **Agent orchestration — LangGraph (`create_react_agent`):** a single tool-calling ReAct agent is the simplest shape that covers all five tools, and LangGraph's Postgres checkpointer gives durable conversation history for free.
- **Tools — `search_archive`, `web_search` (Tavily), `check_recalls` (NHTSA), `update_garage`, `update_instructions`:** Tavily covers the live web the archive can't; NHTSA is the free, keyless, *official* source for recall data; the two memory tools let conversation itself fill the user's profile.
- **Embedding model — OpenAI text-embedding-3-small:** strong price/performance baseline; text-embedding-3-large was A/B-tested in Task 6 and did not provide better performance for the increased cost (~6x).
- **Vector database — Qdrant, in-memory:** the corpus is only 993 chunks, so the serverless function rebuilds the collection from a committed artifact at cold start — no vector-DB service to run, pay for, or keep in sync.
- **Database — Neon Postgres (via Vercel Marketplace, pooled connection):** serverless-friendly Postgres for the LangGraph checkpointer and the three memory tables.
- **Monitoring — LangSmith:** every agent run is traced (reasoning, tool calls, memory writes are auditable), and eval runs land in LangSmith Experiments.
- **Evaluation framework — RAGAS + LangSmith Experiments:** RAG-specific metrics with an LLM judge, logged as comparable experiments (Task 5).
- **User interface — Next.js (React):** streaming chat and the My Garage page, responsive in phone and laptop browsers.
- **Deployment — Vercel:** two seperate projects (frontend + Agent/FastAPI) deployed to Vercel using the Vercel CLI

### Agent workflow

```mermaid
flowchart TD
    U["User message"] --> P["Assemble per-turn system prompt:<br/>base persona + garage profile +<br/>standing instructions + recent session summaries"]
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

### Primary Data source

The corpus is a cleaned CSV export of the MustangDriver.com article archive from WebFlow (their website and content mamagement platform). There are 333 articles in total, covering model history, builds, reviews, mod guides, and buying advice. We also pulled 36 advertisements from the site, including their product information and the affiliate link, so we can make product recommendations to users.

### Chunking strategy

Chunks are created using a hybrid approach. The content comes in a quasi-HTML format from Webflow. We strip out the tags and use the logical seperation boundaries (e.g., a <p> tag) to create chunks. I chose this as the default behavior since content was already logically seperated by the author, and I wanted to preserve this. However, blocks over ~1,000 tokens are recursively split with ~100 tokens of overlap, and fragments under 200 characters merge into a neighbor. This pattern is to prevent chunks from being too large (lower signal-to-noise ratio) or too small (lose the subject and/or surrounding context). Every chunk gets the article title prepended to ensure the subject is preserved. Alongside the embeddings we store metadata (title, live URL, article type, tags, published date) to power citations. Result: **993 chunks across 333 articles** ([`api/ingest.py`](api/ingest.py)).

### External APIs

Two external APIs complement retrieval, selected per-turn by the agent's tool policy:

- **Tavily (live web search):** the archive is a snapshot, so questions about current prices, market values, news, and events can't be answered from it. The agent calls Tavily only when the question is inherently live or archive retrieval comes up empty. RAG is always tried first for Mustang knowledge.
- **NHTSA Recalls API (free, keyless, official):** recall questions are safety-critical, so they bypass both the archive and the web in favor of authoritative government data queried by year/make/model.

## Task 4: End-to-end agentic RAG prototype

The full prototype is built and deployed:

- **Frontend:** https://ask-mustangdriver-web.vercel.app — Next.js streaming chat with markdown rendering, inline archive citations as clickable links, and a **My Garage** page showing everything the assistant has learned (car, mods, wishlist, goals, preferences, recent session summaries).
- **Backend/Agent:** https://ask-mustangdriver-api.vercel.app — FastAPI on Vercel Python Fluid Compute. `POST /chat` streams SSE events (tool notifications, tokens, then a citations payload); `GET /garage/{user_id}` returns the profile, instructions, and summaries; `GET /health` for checks. Full contract in [`api/README.md`](api/README.md).

**Three-type memory model**: 
| Type | What | Where |
|---|---|---|
| **Semantic** | Garage profile: car year/trim/generation, installed mods, wishlist, goals — written by the `update_garage` tool as the user chats | `garage` table (one JSONB row per user, partial updates merge) |
| **Episodic** | Live thread history via LangGraph's Postgres checkpointer, plus a rolling 2–3 sentence per-session summary written by Claude Haiku 4.5 after every turn; recent past-session summaries are injected into the system prompt | Checkpointer tables + `summaries` table |
| **Procedural** | Standing preferences ("keep answers short") written by the `update_instructions` tool, appended to the system prompt every turn | `instructions` table |


## Task 5: Evaluation harness and baseline

### Test data

- **Synthetic set — 51 samples** ([`evals/data/testset.jsonl`](evals/data/testset.jsonl)): generated from the corpus with RAGAS's knowledge-graph testset generation ([`evals/generate_testset.py`](evals/generate_testset.py)).
- **Golden set — 10 hand-written Q/A pairs** ([`evals/data/golden.jsonl`](evals/data/golden.jsonl)): 6 archive-answerable, 2 out-of-archive, 2 recall questions.

### Harness

I use RAGAS to generate synthetic data and evaluate the RAG pipeline. Metrics: **faithfulness, answer relevancy, context precision, context recall** on the synthetic set, plus **answer correctness** on the golden set. The judge is **GPT-5-mini** (different family than the agent, to prevent self-grading). Every run logs to **LangSmith Experiments** against the `ask-mustangdriver-synthetic` / `ask-mustangdriver-golden` datasets.

### Baseline results

| Variant | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness |
|---|---:|---:|---:|---:|---:|
| baseline (dense top-5, 3-small) | 0.9187 | 0.6358 | 0.7564 | 0.6638 | 0.6473 |

### Conclusions from the baseline

- **Generation is not the problem.** Faithfulness of 0.92 means that when relevant context is retrieved, the model grounds its answer in it rather than inventing/hallucinating.
- **Retrieval is the bottleneck.** Context recall of 0.66 says dense retrieval misses roughly a third of the context needed for a complete answer.
- **Answer correctness is relatively low** However, in a manual review of the answers, they seem to fully satisfy the question and, while the full answer text is not exact, the core answer appears to closely match that of the ground truth.

## Task 6: Improving the prototype

### Advanced retrieval: hybrid BM25 + dense with reciprocal rank fusion

Mustang jargon is **exact-match heavy**, e.g., chassis codes (S550, SN95), model names (GT350, Cobra Jet, Dark Horse). Hybrid retrieval keeps dense's semantic search functions while letting BM25 pick up on the exact tokens. BM25 and dense rankings are fused with reciprocal rank fusion (k=60) over each ranker's result.

### Second change: embedding model A/B

I also experimented using small vs. large embeddings on the corpus (`text-embedding-3-small` → `text-embedding-3-large`).

### Results (full table, from [`evals/results/comparison.md`](evals/results/comparison.md))

| Variant | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness |
|---|---:|---:|---:|---:|---:|
| baseline (dense, 3-small) | 0.9187 | 0.6358 | 0.7564 | 0.6638 | 0.6473 |
| **hybrid RRF (3-small)** | **0.9301** | **0.6560** | 0.7907 | **0.7426** | 0.6422 |
| dense (3-large) | 0.9158 | 0.5897 | 0.7645 | 0.6427 | 0.6398 |
| hybrid RRF (3-large) | 0.9278 | 0.6508 | **0.7955** | 0.6893 | 0.6316 |

All 8 runs (4 variants × synthetic/golden) are logged as LangSmith Experiments.

### Conclusion: what improved and why

**Hybrid RRF on 3-small embeddings** improved the context recall (0.664 → 0.743) and context precision (0.756 → 0.791). From these results, we can conclude that BM25's exact-token matching recovers jargon-keyed chunks that dense embeddings miss, and RRF folds them in without sacrificing dense's semantic matches. 

**The bigger embedding model provided no improvement** Dense 3-large is a wash versus 3-small at a ~6x higher cost. On this corpus's specialized vocabulary, lexical matching adds real signal; more embedding dimensions don't.

**Answer correctness is relatively flat across all four variants** (0.63–0.65) - Although the scores seem low, when comparing even lower scoring golden dataset answers with the ground truth, the answers are appear appropriate and accurate. I believe the lower scoring is due to variance in tone, response style, and differing verbosity. In future iteration, I plan to experiment with this to ensure the agent is more consistent.

**Decision: ship hybrid RRF + text-embedding-3-small.**

## Task 7: Next steps for Demo Day

### Reflection
For Demo Day, I plan on keeping most of the current implementation, although it certainly needs refinement. The RAG pipeline is relatively robust - it consults the knowledge base, and augments with web search when necessary. The mechanism has been gamified to make a fun, interactive car statistics visualization for the user. I think this is a good addition to drive engagement, while simultaneously functioning a customer profile to provide helpful, targeted product recommendations. The app reliably calls tools when it needs to save information about the user's installed car upgrades, upgrades that they hope to one day install (wish list), or communication preferences. The agent also stays within its guardrails well, successfully rejecting to answer questions that are irrelevant to its scope.

For Demo Day, I plan to improve the answer correctness metric versus our golden dataset. I believe the current gaps are due to differences in tone: the golden answers are pretty straight-forward and to-the-point, while the default tone of the model is more verbose and playful. I need to consult with the business users to align on an ideal tone for the agent, and align the golden answers more closely to that. Additionally, I want to look into the other RAG metrics, such as context recall and precision. I believe these can be further improved with more RAG tuning.

### Loom


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

### Optional Google sign-in

Login is optional: anonymous browser-UUID users keep working with no login. Signing in with Google binds the browser's current anonymous `user_id` to the Google account (identities table, `google_sub → user_id`), so the same garage/cars/chats follow the user across devices. Known limit: anonymous data accumulated on a *second* device before logging in there is orphaned when that device switches to the canonical id (no merge).

Config (feature is hidden until set):

- `GOOGLE_CLIENT_ID` (API, `.env`) — the Google OAuth Client ID; verify audience for ID tokens
- `NEXT_PUBLIC_GOOGLE_CLIENT_ID` (web, same value) — renders the Sign in button
- `AUTH_JWT_SECRET` (API, `.env`) — signs the app session JWT; any long random string (`openssl rand -hex 32`)

Owner action: in Google Cloud Console create an **OAuth Client ID (Web application)** with authorized JavaScript origins `http://localhost:3000` and `https://ask-mustangdriver-web.vercel.app`, then set the three variables above (locally and in the Vercel projects).

## Deploying to production

One command, after the show-first ship gate (local build + screenshots + owner approval):

```sh
./deploy.sh
```

It deploys the API to Vercel production, health-checks `/health`, then deploys the web app and smoke-checks it — failing fast at each step. Needs a logged-in Vercel CLI and Node 18+ (when the shell default is older, it picks up the newest nvm-installed Node automatically). Both apps are already linked (`api/.vercel`, `web/.vercel`); `vercel link` re-creates the links if lost. Production URLs: [web](https://ask-mustangdriver-web.vercel.app) · [API](https://ask-mustangdriver-api.vercel.app).

## Repo layout

```
deploy.sh       production deploy: api → health check → web → smoke check
AGENTS.md       operating guide for coding agents (CLAUDE.md symlinks to it)
api/            FastAPI backend: agent, tools, memory, ingestion, tests
  index_artifact/   committed chunks + vectors (the corpus ships pre-embedded)
  ads_artifact/     committed advertiser product catalog + vectors
  EXCLUDED_ARTICLES.md   promo-exclusion rule + full list of 21
evals/          RAGAS harness, datasets, experiments notebook, results
  results/          per-variant aggregates + comparison.md
web/            Next.js frontend: streaming chat + My Garage
requirements/   the certification challenge brief this README answers
docker-compose.yml   local Postgres
.env.example    required environment variables
```

**Monitoring:** all agent runs and eval experiments are in LangSmith under the project `ask-mustangdriver` (traces show reasoning, tool calls, and memory writes; Experiments hold the 8 eval runs behind the Task 6 table).
