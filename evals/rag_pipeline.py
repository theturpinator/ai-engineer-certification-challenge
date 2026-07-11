"""The RAG pipeline under test, isolated from the production agent (api/app.py):
top-k retrieval (dense, or hybrid BM25+dense with RRF) over the committed index
artifact + generation with a FIXED prompt. Parameterized (retriever / top_k /
embedding_model / chat_model) so ticket #11 variants can reuse it.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"
ARTIFACT = REPO / "api" / "index_artifact"
COLLECTION = "archive"
RRF_K = 60
RRF_POOL = 50  # fuse top-50 from each ranking; deeper ranks contribute noise, not signal

# Chunk vectors per embedding model, same row order as chunks.jsonl.
# vectors_3large.npz is built by experiments.ipynb (ticket #11).
VECTOR_FILES = {
    "openai/text-embedding-3-small": ARTIFACT / "vectors.npz",
    "openai/text-embedding-3-large": Path(__file__).parent / "data" / "vectors_3large.npz",
}


def tokenize(text: str) -> list[str]:
    """Lowercase; keep decimal numbers ("5.0") and alnum runs ("s550") whole."""
    return re.findall(r"\d+(?:\.\d+)+|[a-z0-9]+", text.lower())

# Fixed prompt: mirrors the production tool policy (ground in archive, admit
# gaps, route recall questions to NHTSA) without the agent's tools/memory.
RAG_PROMPT = """You are the Ask MustangDriver assistant. Answer the question \
using ONLY the context below, drawn from the MustangDriver.com article archive.

- If the context does not contain the information needed, say the archive \
doesn't appear to cover it — do not guess.
- If the question asks about vehicle safety recalls, say that recall questions \
need the official NHTSA recall lookup, which this archive search cannot answer.

Context:
{context}

Question: {question}"""


def gateway_chat(model: str, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=GATEWAY_URL,
        api_key=os.environ["AI_GATEWAY_API_KEY"],
        **kwargs,
    )


def gateway_embeddings(model: str = "openai/text-embedding-3-small") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model,
        base_url=GATEWAY_URL,
        api_key=os.environ["AI_GATEWAY_API_KEY"],
        check_embedding_ctx_length=False,  # gateway wants raw strings, not token arrays
    )


class RagPipeline:
    def __init__(
        self,
        retriever: str = "dense",
        top_k: int = 5,
        embedding_model: str = "openai/text-embedding-3-small",
        chat_model: str = "anthropic/claude-sonnet-4.5",
    ):
        if retriever not in ("dense", "hybrid"):
            raise ValueError(f"unknown retriever {retriever!r}")
        if embedding_model not in VECTOR_FILES:
            raise ValueError(f"no chunk vectors for {embedding_model!r}")
        self.retriever = retriever
        self.top_k = top_k
        self._embeddings = gateway_embeddings(embedding_model)
        self._llm = gateway_chat(chat_model)
        with open(ARTIFACT / "chunks.jsonl") as f:
            self._chunks = [json.loads(line) for line in f]
        vectors = np.load(VECTOR_FILES[embedding_model])["vectors"]
        self._qdrant = QdrantClient(location=":memory:")
        self._qdrant.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        self._qdrant.upload_collection(
            COLLECTION, vectors=vectors, payload=self._chunks, ids=list(range(len(self._chunks)))
        )
        if retriever == "hybrid":
            self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self._chunks])

    async def retrieve(self, question: str) -> list[dict]:
        """Top-k over the archive (dense, or BM25+dense RRF); returns chunk payloads."""
        vector = await self._embeddings.aembed_query(question)
        limit = RRF_POOL if self.retriever == "hybrid" else self.top_k
        hits = self._qdrant.query_points(COLLECTION, query=vector, limit=limit).points
        if self.retriever == "dense":
            return [h.payload for h in hits]
        # Reciprocal rank fusion (k=60) of the dense and BM25 top-50 lists.
        # Point ids are row indices into chunks.jsonl, so both rankings share ids.
        scores = self._bm25.get_scores(tokenize(question))
        bm25_ids = [int(i) for i in np.argsort(-scores)[:RRF_POOL] if scores[i] > 0]
        fused: dict[int, float] = defaultdict(float)
        for ranking in ([h.id for h in hits], bm25_ids):
            for rank, doc_id in enumerate(ranking):
                fused[doc_id] += 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.__getitem__, reverse=True)[: self.top_k]
        return [self._chunks[i] for i in top]

    async def answer(self, question: str, contexts: list[dict]) -> str:
        context = "\n\n---\n\n".join(c["text"] for c in contexts)
        resp = await self._llm.ainvoke(
            RAG_PROMPT.format(context=context, question=question)
        )
        return resp.content
