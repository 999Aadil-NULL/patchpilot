"""tree-sitter chunks → FastEmbed → Qdrant(":memory:") semantic ranking.

Memory-conscious (WSL default ~7 GB shared with Docker + Ollama): small batches, capped threads,
1000-char chunk text, and a per-file on-disk embedding cache so reruns don't re-embed.
"""
from __future__ import annotations

import hashlib
import os
import threading
import uuid
from pathlib import Path

import numpy as np

from agent.config import CACHE_DIR
from agent.localize.repomap import chunk_file
from agent.schemas import CodeChunk

PRIMARY_MODEL = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_CHARS = 1000      # chunk text truncated for embedding (speed + memory; ranking quality held up)
EMBED_BATCH = 8         # batch 16 OOM-killed the process at ~3.5 GB RSS on 7 GB WSL
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "6"))
EMB_CACHE = CACHE_DIR / "emb"

_model = None
_model_name = ""
_lock = threading.Lock()


def _embedder():
    global _model, _model_name
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding

            try:
                _model, _model_name = TextEmbedding(PRIMARY_MODEL, threads=EMBED_THREADS), PRIMARY_MODEL
            except Exception:
                _model, _model_name = TextEmbedding(FALLBACK_MODEL, threads=EMBED_THREADS), FALLBACK_MODEL
        return _model


def _doc_text(c: CodeChunk) -> str:
    return f"{c.file_path} :: {c.symbol}\n{c.code[:EMBED_CHARS]}"


def _embed_file(chunks: list[CodeChunk]) -> np.ndarray:
    model = _embedder()
    docs = [_doc_text(c) for c in chunks]
    key = hashlib.sha256(("\x00".join([_model_name, str(EMBED_CHARS)] + docs)).encode()).hexdigest()
    fp = EMB_CACHE / _model_name.replace("/", "__") / f"{key}.npy"
    if fp.exists():
        try:
            return np.load(fp)
        except Exception:
            pass
    vecs = np.array(list(model.embed(docs, batch_size=EMBED_BATCH)), dtype=np.float32)
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(fp, vecs)
    return vecs


def rank_chunks(repo_dir: Path, files: list[str], query: str, top_k: int = 8) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    blocks: list[np.ndarray] = []
    for f in files:
        fc = chunk_file(repo_dir, f)
        if fc:
            chunks.extend(fc)
            blocks.append(_embed_file(fc))
    if not chunks:
        return []
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    vecs = np.concatenate(blocks)
    qvec = np.array(list(_embedder().embed([query[:EMBED_CHARS * 2]]))[0], dtype=np.float32)

    client = QdrantClient(":memory:")
    name = f"chunks_{uuid.uuid4().hex[:8]}"
    client.create_collection(name, vectors_config=VectorParams(size=len(qvec), distance=Distance.COSINE))
    client.upsert(name, points=[PointStruct(id=i, vector=v.tolist(), payload={"i": i}) for i, v in enumerate(vecs)])
    hits = client.query_points(name, query=qvec.tolist(), limit=top_k).points
    client.close()

    out = []
    for h in hits:
        c = chunks[h.payload["i"]].model_copy()
        c.score = round(float(h.score), 4)
        out.append(c)
    return out
