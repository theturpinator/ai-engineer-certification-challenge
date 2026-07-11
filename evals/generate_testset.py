"""One-time: generate the ~50-sample RAGAS synthetic testset from the corpus.

Groups the index-artifact chunks back into whole articles, samples a subset
(seeded, so reruns see the same corpus slice), builds the ragas knowledge
graph, and generates questions. Persists to data/testset.jsonl so evals are
reproducible without regenerating.

Run: .venv/bin/python generate_testset.py
"""

import json
import random
from pathlib import Path

from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms

from rag_pipeline import ARTIFACT, gateway_chat, gateway_embeddings

TESTSET_SIZE = 50
N_ARTICLES = 60  # ponytail: KG transforms cost ~5 LLM calls/article; 60 is plenty for 50 Qs
OUT = Path(__file__).parent / "data" / "testset.jsonl"
KG_CACHE = Path(__file__).parent / "data" / "kg.json"  # transforms checkpoint; delete to rebuild


def load_articles() -> list[Document]:
    """Rebuild whole articles from chunks (chunk text repeats the title prefix)."""
    articles: dict[str, dict] = {}
    with open(ARTIFACT / "chunks.jsonl") as f:
        for line in f:
            c = json.loads(line)
            a = articles.setdefault(c["url"], {"title": c["title"], "parts": []})
            text = c["text"]
            if text.startswith(c["title"]):
                text = text[len(c["title"]):].lstrip()
            a["parts"].append(text)
    docs = [
        Document(
            page_content=a["title"] + "\n\n" + "\n".join(a["parts"]),
            metadata={"title": a["title"], "url": url},
        )
        for url, a in articles.items()
    ]
    docs = [d for d in docs if len(d.page_content) > 1500]  # too-short docs get KG-filtered anyway
    random.Random(42).shuffle(docs)
    return docs[:N_ARTICLES]


def main():
    docs = load_articles()
    print(f"{len(docs)} articles sampled for generation")
    # Sonnet generates; the judge in run_eval.py stays gpt-5-mini (different family).
    llm = LangchainLLMWrapper(gateway_chat("anthropic/claude-sonnet-4.5"))
    emb = LangchainEmbeddingsWrapper(gateway_embeddings())

    if KG_CACHE.exists():
        kg = KnowledgeGraph.load(KG_CACHE)
        print(f"loaded cached knowledge graph ({len(kg.nodes)} nodes)")
    else:
        kg = KnowledgeGraph(
            nodes=[
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": d.page_content,
                        "document_metadata": d.metadata,
                    },
                )
                for d in docs
            ]
        )
        # max_workers=8: the default 16-way fan-out crashed the process locally
        apply_transforms(
            kg,
            default_transforms(documents=docs, llm=llm, embedding_model=emb),
            run_config=RunConfig(max_workers=8),
        )
        KG_CACHE.parent.mkdir(exist_ok=True)
        kg.save(KG_CACHE)
    generator = TestsetGenerator(llm=llm, embedding_model=emb, knowledge_graph=kg)

    try:
        testset = generator.generate(testset_size=TESTSET_SIZE)
    except Exception as e:
        # Multi-hop synthesizers need KG clusters that prose corpora don't always
        # yield; fall back to single-hop-only without redoing the transforms.
        print(f"default query distribution failed ({e}); falling back to single-hop only")
        from ragas.testset.synthesizers.single_hop.specific import (
            SingleHopSpecificQuerySynthesizer,
        )
        testset = generator.generate(
            testset_size=TESTSET_SIZE,
            query_distribution=[(SingleHopSpecificQuerySynthesizer(llm=llm), 1.0)],
        )

    df = testset.to_pandas()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for _, row in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "question": row["user_input"],
                        "reference": row["reference"],
                        "reference_contexts": list(row["reference_contexts"]),
                        "synthesizer": row["synthesizer_name"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(df)} samples to {OUT}")


if __name__ == "__main__":
    main()
