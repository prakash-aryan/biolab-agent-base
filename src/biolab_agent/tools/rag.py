"""retrieve_protocol tool  -  RAG over Qdrant collection ``protocols``."""

from __future__ import annotations

import functools

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from biolab_agent.config import load_settings
from biolab_agent.schemas import ProtocolHit

COLLECTION = "protocols"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@functools.lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def _client() -> QdrantClient:
    settings = load_settings()
    return QdrantClient(url=settings.qdrant_url, timeout=10.0)


def retrieve_protocol(query: str, k: int = 5) -> list[ProtocolHit]:
    """Return the top-``k`` most similar protocol chunks for ``query``."""
    if not query.strip():
        return []
    vec = _embedder().encode([query], normalize_embeddings=True).tolist()[0]

    client = _client()
    # qdrant-client >=1.12 deprecates `search(...)` in favor of `query_points(...)`.
    resp = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=max(1, min(k, 25)),
        with_payload=True,
    )
    out: list[ProtocolHit] = []
    for h in resp.points:
        payload = h.payload or {}
        out.append(
            ProtocolHit(
                doc_id=str(payload.get("doc_id", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                title=str(payload.get("title", "")),
                source_url=payload.get("source_url"),
                text=str(payload.get("text", ""))[:1200],
                score=float(h.score),
            )
        )
    return out


retrieve_protocol_spec = {
    "type": "function",
    "function": {
        "name": "retrieve_protocol",
        "description": (
            "Search the local protocol library (indexed from OpenTrons) and "
            "return the top-k most relevant protocol chunks with their doc_id, "
            "title, source URL, and a text excerpt. Use this when the user asks "
            "for a procedure, SOP, or reference protocol."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what to find.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many hits to return (1-10).",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}
