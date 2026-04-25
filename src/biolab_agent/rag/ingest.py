"""Reference corpus ingestion  -  chunk, embed, index into Qdrant.

Baseline so retrieval strategy is the interesting part to build. Replace
with a better chunker, a different embedding model, or a different vector
DB as needed.

The corpus lives at ``data/protocols/*.jsonl`` with one document per line::

    {"doc_id": "...", "title": "...", "source_url": "...", "text": "..."}
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from biolab_agent.config import load_settings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    source_url: str | None
    text: str


def _simple_chunks(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Character-window chunker with overlap."""
    if not text:
        return []
    if len(text) <= size:
        return [text.strip()]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def iter_documents(corpus_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield one document dict per non-empty JSONL line in ``corpus_dir``."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %s:%d  -  %s", path, i, exc)


def iter_chunks(corpus_dir: Path) -> Iterator[Chunk]:
    """Stream per-chunk records from the corpus."""
    for doc in iter_documents(corpus_dir):
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        for ci, ctext in enumerate(_simple_chunks(doc.get("text", ""))):
            yield Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:{ci}",
                title=doc.get("title", doc_id),
                source_url=doc.get("source_url"),
                text=ctext,
            )


# ----------------------------------------------------------------------------
# Embedding + indexing
# ----------------------------------------------------------------------------


def _load_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def ingest_corpus(
    corpus_dir: Path,
    collection: str = "protocols",
    embed_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Chunk + embed + upsert the corpus into a Qdrant collection.

    The collection is recreated every run  -  cheap for the starter corpus.
    """
    settings = load_settings()

    chunks = list(iter_chunks(corpus_dir))
    if not chunks:
        return {"chunks": 0, "documents": 0, "collection": collection, "dry_run": dry_run}

    stats: dict[str, Any] = {
        "chunks": len(chunks),
        "documents": len({c.doc_id for c in chunks}),
        "collection": collection,
        "dry_run": dry_run,
        "embed_model": embed_model,
    }

    if dry_run:
        return stats

    logger.info("Loading embedder %s", embed_model)
    embedder = _load_embedder(embed_model)
    dim = embedder.get_sentence_embedding_dimension()

    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    client = QdrantClient(url=settings.qdrant_url)

    # Recreate the collection so re-runs stay idempotent. `recreate_collection`
    # was deprecated in qdrant-client 1.12; do it in two steps.
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    points: list[qm.PointStruct] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embedder.encode(
            [c.text for c in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for chunk, vec in zip(batch, vectors, strict=True):
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector=vec.tolist(),
                    payload={
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "source_url": chunk.source_url,
                        "text": chunk.text,
                    },
                )
            )

    client.upsert(collection_name=collection, points=points)
    stats["vector_dim"] = dim
    stats["upserted"] = len(points)
    return stats
