"""The RAG pipeline under test, isolated from the production agent (api/app.py):
dense top-k retrieval over the committed index artifact + generation with a
FIXED prompt. Parameterized (retriever / top_k / embedding_model / chat_model)
so ticket #11 variants can reuse it.
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"
ARTIFACT = REPO / "api" / "index_artifact"
COLLECTION = "archive"

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
        # ponytail: "dense" is the only retriever until ticket #11 adds hybrid.
        # NB: committed vectors.npz is text-embedding-3-small; a different
        # embedding_model requires re-embedding the corpus (#11's job).
        if retriever != "dense":
            raise ValueError(f"unknown retriever {retriever!r}")
        self.top_k = top_k
        self._embeddings = gateway_embeddings(embedding_model)
        self._llm = gateway_chat(chat_model)
        with open(ARTIFACT / "chunks.jsonl") as f:
            chunks = [json.loads(line) for line in f]
        vectors = np.load(ARTIFACT / "vectors.npz")["vectors"]
        self._qdrant = QdrantClient(location=":memory:")
        self._qdrant.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        self._qdrant.upload_collection(
            COLLECTION, vectors=vectors, payload=chunks, ids=list(range(len(chunks)))
        )

    async def retrieve(self, question: str) -> list[dict]:
        """Dense top-k over the archive; returns chunk payloads (text/title/url)."""
        vector = await self._embeddings.aembed_query(question)
        hits = self._qdrant.query_points(COLLECTION, query=vector, limit=self.top_k).points
        return [h.payload for h in hits]

    async def answer(self, question: str, contexts: list[dict]) -> str:
        context = "\n\n---\n\n".join(c["text"] for c in contexts)
        resp = await self._llm.ainvoke(
            RAG_PROMPT.format(context=context, question=question)
        )
        return resp.content
