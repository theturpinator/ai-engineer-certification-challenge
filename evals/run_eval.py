"""Baseline eval, one command: runs the RAG pipeline over the synthetic testset
(faithfulness, answer relevancy, context precision, context recall) and the
golden set (answer correctness), judged by gpt-5-mini via the gateway, logged
to LangSmith Experiments.

Run: .venv/bin/python run_eval.py [--experiment baseline-dense]
"""

import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path

from langsmith import Client, aevaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerSimilarity,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from rag_pipeline import RagPipeline, gateway_chat, gateway_embeddings

HERE = Path(__file__).parent
JUDGE_MODEL = "openai/gpt-5-mini"  # different family than the Sonnet agent
SYNTHETIC_DS = "ask-mustangdriver-synthetic"
GOLDEN_DS = "ask-mustangdriver-golden"


def judge_metrics() -> dict:
    emb = LangchainEmbeddingsWrapper(gateway_embeddings())
    # One LLM instance per metric: ragas mutates llm.temperature/.n per call,
    # so sharing one across concurrently-running metrics is a race.
    def llm():
        return LangchainLLMWrapper(gateway_chat(JUDGE_MODEL))

    return {
        # strictness=1: the gateway silently ignores n>1 (returns 1 completion),
        # which breaks ResponseRelevancy's default 3-question generation.
        "faithfulness": Faithfulness(llm=llm()),
        "answer_relevancy": ResponseRelevancy(llm=llm(), embeddings=emb, strictness=1),
        "context_precision": LLMContextPrecisionWithReference(llm=llm()),
        "context_recall": LLMContextRecall(llm=llm()),
        # answer_similarity set explicitly: ragas only wires it up inside its own
        # evaluate() init, and we call single_turn_ascore directly.
        "answer_correctness": AnswerCorrectness(
            llm=llm(), embeddings=emb, answer_similarity=AnswerSimilarity(embeddings=emb)
        ),
    }


def ragas_evaluator(name: str, metric):
    async def _eval(run, example):
        sample = SingleTurnSample(
            user_input=example.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["contexts"],
            reference=example.outputs["reference"],
        )
        try:
            score = await metric.single_turn_ascore(sample)
        except Exception as e:  # nan-skip: a few judge failures shouldn't kill the run
            return {"key": name, "score": None, "comment": f"error: {e}"[:500]}
        if score is None or math.isnan(score):
            return {"key": name, "score": None, "comment": "nan"}
        return {"key": name, "score": float(score)}

    _eval.__name__ = name
    return _eval


def ensure_dataset(client: Client, name: str, rows: list[dict]):
    """Create the LangSmith dataset once; reuse on later runs."""
    if client.has_dataset(dataset_name=name):
        return
    ds = client.create_dataset(name)
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"question": r["question"]},
                "outputs": {k: v for k, v in r.items() if k != "question"},
            }
            for r in rows
        ],
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


async def run(experiment: str, only: str | None):
    client = Client()

    synthetic = [
        {"question": r["question"], "reference": r["reference"], "synthesizer": r["synthesizer"]}
        for r in load_jsonl(HERE / "data" / "testset.jsonl")
    ]
    golden = [
        {"question": r["question"], "reference": r["reference_answer"], "category": r["category"]}
        for r in load_jsonl(HERE / "data" / "golden.jsonl")
    ]
    ensure_dataset(client, SYNTHETIC_DS, synthetic)
    ensure_dataset(client, GOLDEN_DS, golden)

    pipeline = RagPipeline()  # baseline dense config

    async def target(inputs: dict) -> dict:
        contexts = await pipeline.retrieve(inputs["question"])
        answer = await pipeline.answer(inputs["question"], contexts)
        return {"answer": answer, "contexts": [c["text"] for c in contexts]}

    metrics = judge_metrics()
    out = HERE / "results" / "baseline.json"
    report = json.loads(out.read_text()) if out.exists() else {}  # --only reruns merge in
    report.update({"judge": JUDGE_MODEL, "config": {"retriever": "dense", "top_k": pipeline.top_k}})

    for dataset, metric_names, suffix in [
        (SYNTHETIC_DS, ["faithfulness", "answer_relevancy", "context_precision", "context_recall"], "synthetic"),
        (GOLDEN_DS, ["answer_correctness"], "golden"),
    ]:
        if only and suffix != only:
            continue
        results = await aevaluate(
            target,
            data=dataset,
            evaluators=[ragas_evaluator(n, metrics[n]) for n in metric_names],
            experiment_prefix=f"{experiment}-{suffix}",
            max_concurrency=8,
            client=client,
        )
        scores, errors = defaultdict(list), defaultdict(int)
        n_rows = 0
        async for row in results:
            n_rows += 1
            for r in row["evaluation_results"]["results"]:
                if r.score is None:
                    errors[r.key] += 1
                else:
                    scores[r.key].append(r.score)
        project = client.read_project(project_name=results.experiment_name)
        report[suffix] = {
            "experiment": results.experiment_name,
            "url": str(project.url),
            "samples": n_rows,
            "metrics": {
                k: {
                    "mean": round(sum(scores[k]) / len(scores[k]), 4) if scores[k] else None,
                    "scored": len(scores[k]),
                    "failed": errors[k],
                }
                for k in metric_names
            },
        }
        print(f"\n=== {suffix} ({n_rows} samples) — {results.experiment_name}")
        print(project.url)
        for k, v in report[suffix]["metrics"].items():
            mean = f"{v['mean']:.4f}" if v["mean"] is not None else "ALL FAILED"
            print(f"  {k:20s} {mean}  ({v['scored']} scored, {v['failed']} failed)")

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline-dense")
    parser.add_argument("--only", choices=["synthetic", "golden"], default=None)
    args = parser.parse_args()
    asyncio.run(run(args.experiment, args.only))
