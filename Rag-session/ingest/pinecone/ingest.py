"""Ingest embeddings into Pinecone vector index.

Batch upsert: 100 vectors per call.
Metadata: text truncated to 1000 chars (40KB limit).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

BATCH_SIZE = 100
TEXT_LIMIT = 1000  # metadata text truncation
DIM = 4096


def _get_required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing. Set it in the .env file.")
    return value


def ingest(progress_callback=None):
    """Batch upsert embeddings into Pinecone vector index.

    Args:
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        int: Number of vectors upserted.

    Hints:
        - Load embeddings from PROCESSED_DIR / "embeddings.npy"
        - Load IDs from PROCESSED_DIR / "embedding_ids.json"
        - Load texts from RAW_DIR / "corpus.jsonl" for metadata
        - Connect: Pinecone(api_key=...) → pc.Index(index_name)
        - Upsert format: {"id": ..., "values": [...], "metadata": {"text": ...}}
        - Batch size: BATCH_SIZE (100), truncate text to TEXT_LIMIT (1000) chars
    """
    embeddings_path = PROCESSED_DIR / "embeddings.npy"
    ids_path = PROCESSED_DIR / "embedding_ids.json"
    corpus_path = RAW_DIR / "corpus.jsonl"
    for path in (embeddings_path, ids_path, corpus_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    ids = [str(doc_id) for doc_id in json.loads(ids_path.read_text(encoding="utf-8"))]
    if embeddings.shape != (len(ids), DIM):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match ({len(ids)}, {DIM})."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("embedding_ids.json contains duplicate document IDs.")

    texts: dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                document = json.loads(line)
                texts[str(document["id"])] = str(document["text"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid corpus record at line {line_number}."
                ) from exc
    missing_ids = [doc_id for doc_id in ids if doc_id not in texts]
    if missing_ids:
        raise ValueError(
            f"Corpus is missing {len(missing_ids)} embedding IDs (example: {missing_ids[0]})."
        )

    pc = Pinecone(api_key=_get_required_env("PINECONE_API_KEY"))
    index_name = (os.getenv("PINECONE_INDEX") or "ragsession").strip()
    existing_names = set(pc.list_indexes().names())
    if index_name not in existing_names:
        pc.create_index(
            name=index_name,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=(os.getenv("PINECONE_CLOUD") or "aws").strip(),
                region=(os.getenv("PINECONE_REGION") or "us-east-1").strip(),
            ),
        )
        # Index creation is asynchronous. Wait briefly before opening the data plane.
        for _ in range(60):
            description = pc.describe_index(index_name)
            status = description.status
            ready = status.get("ready", False) if isinstance(status, dict) else status.ready
            if ready:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Pinecone index '{index_name}' was not ready in time.")
    else:
        description = pc.describe_index(index_name)
        if int(description.dimension) != DIM:
            raise ValueError(
                f"Pinecone index '{index_name}' has dimension {description.dimension}; "
                f"expected {DIM}."
            )
        if str(description.metric) != "cosine":
            raise ValueError(
                f"Pinecone index '{index_name}' uses {description.metric}; expected cosine."
            )

    index = pc.Index(index_name)
    total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    upserted = 0
    for batch_number, start in enumerate(range(0, len(ids), BATCH_SIZE), 1):
        end = min(start + BATCH_SIZE, len(ids))
        vectors = [
            {
                "id": ids[idx],
                "values": embeddings[idx].tolist(),
                "metadata": {"text": texts[ids[idx]][:TEXT_LIMIT]},
            }
            for idx in range(start, end)
        ]
        index.upsert(vectors=vectors)
        upserted += len(vectors)
        if progress_callback:
            progress_callback(batch_number, total_batches)

    return upserted


if __name__ == "__main__":
    ingest()
