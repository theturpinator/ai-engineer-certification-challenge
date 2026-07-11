# Evals

Scores the RAG pipeline (retrieval over `api/index_artifact/` + fixed-prompt
generation with `anthropic/claude-sonnet-4.5`) in isolation from the production
agent, so retrieval/prompt changes can be measured. Judge is
`openai/gpt-5-mini` (different family than the agent); everything goes through
the Vercel AI Gateway using `AI_GATEWAY_API_KEY` from the repo-root `.env`.

## Setup

```sh
cd evals
uv venv --python 3.13
uv pip install -r requirements.txt
```

## Run

```sh
# 1. Generate the ~50-sample synthetic testset (one-time; already committed to data/testset.jsonl)
.venv/bin/python generate_testset.py

# 2. Run the baseline eval: scores + LangSmith experiments + results/baseline.json
.venv/bin/python run_eval.py            # --experiment baseline-dense (default)
```

`--only synthetic|golden` reruns one half and merges it into `results/baseline.json`.

`run_eval.py` runs the pipeline over both datasets and logs two LangSmith
experiments (`<experiment>-synthetic`, `<experiment>-golden`) against the
`ask-mustangdriver-synthetic` / `ask-mustangdriver-golden` datasets (created on
first run):

- synthetic testset: faithfulness, answer relevancy, context precision, context recall
- golden set (`data/golden.jsonl`): answer correctness

Golden entries are marked `"status": "stub"` — reference answers are the
author's best stubs, to be refined. Categories: `archive` (answerable),
`trap` (correct behavior = admit the archive doesn't cover it), `recall`
(correct behavior = route to the official NHTSA recall lookup).

## Retrieval experiments (issue #11)

`experiments.ipynb` reruns the full harness against four pipeline variants and
produces one combined results table (also saved to `results/comparison.md`):

- **baseline** — dense top-5, `text-embedding-3-small` (loaded from the
  committed `results/baseline.json`, never rerun)
- **hybrid-rrf** — BM25 (`rank_bm25`, lowercase alnum tokens with decimals kept
  whole so "S550"/"5.0" survive) + dense, fused with reciprocal rank fusion
  (k=60) over the two top-50 lists
- **dense-3large** — index re-embedded with `text-embedding-3-large`
  (`data/vectors_3large.npz`, built by the notebook, committed)
- **hybrid-3large** — both combined

Run it headless (each non-baseline variant is 51+10 samples, ~10-25 min):

```sh
.venv/bin/jupyter-nbconvert --to notebook --execute --inplace experiments.ipynb --ExecutePreprocessor.timeout=-1
```

Each variant's aggregates are cached in `results/<variant>.json` (same shape as
`baseline.json`) and its runs are logged to LangSmith Experiments prefixed with
the variant name (e.g. `hybrid-rrf-synthetic-…`). Delete a results file to
rerun that variant. To eval a new variant, add a `RagPipeline` kwargs entry to
the `VARIANTS` dict in the notebook.

## Gateway gotchas (learned the hard way)

- `openai/gpt-5-mini` is a reasoning model; the gateway tolerates the
  `temperature` values ragas sets (strips them), but **silently ignores `n>1`**
  (returns one completion). `ResponseRelevancy` therefore runs with
  `strictness=1`, and each metric gets its own LLM instance since ragas mutates
  `llm.temperature`/`.n` per call.
- ragas 0.2.x needs the langchain 0.3 line — see `requirements.txt` pins.
- Gateway embeddings need `check_embedding_ctx_length=False` (raw strings).
- `AnswerCorrectness` needs `answer_similarity` passed explicitly when scoring
  via `single_turn_ascore` (ragas only wires it inside its own `evaluate()`).
- KG transforms crashed the process at the default 16-way concurrency;
  `generate_testset.py` uses `max_workers=8` and checkpoints the knowledge
  graph to `data/kg.json` (gitignored; delete to rebuild).
