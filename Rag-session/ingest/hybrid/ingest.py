"""Ingest corpus into Elasticsearch Hybrid index (wiki-hybrid).

Index mapping: text field + dense_vector(4096, cosine).
Bulk chunk_size=100 (heavier with 4096-dim vectors).
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

load_dotenv()

INDEX_NAME = "wiki-hybrid"
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

INDEX_MAPPINGS = {
    "properties": {
        "text": {"type": "text", "analyzer": "standard"},
        "embedding": {
            "type": "dense_vector",
            "dims": 4096,
            "index": True,
            "similarity": "cosine",
        },
    }
}


def get_es_client() -> Elasticsearch:
    endpoint = (os.getenv("ELASTIC_ENDPOINT") or "").strip()
    api_key = (os.getenv("ELASTIC_API_KEY") or "").strip()
    if not endpoint or not api_key:
        raise RuntimeError(
            "Elasticsearch credentials are missing. Set ELASTIC_ENDPOINT and "
            "ELASTIC_API_KEY in the .env file."
        )
    return Elasticsearch(
        endpoint,
        api_key=api_key,
        request_timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )


def _generate_actions(corpus_path: Path, embeddings: np.ndarray, ids: list[str]):
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["id"]
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            yield {
                "_index": INDEX_NAME,
                "_id": doc_id,
                "_source": {
                    "text": doc["text"],
                    "embedding": embeddings[idx].tolist(),
                },
            }


def ingest(progress_callback=None):
    """Create hybrid index (text + dense_vector) and bulk-ingest corpus.

    Args:
        progress_callback: Optional callback(count) called after completion.

    Returns:
        int: Number of documents indexed.

    Hints:
        - Load embeddings from PROCESSED_DIR / "embeddings.npy"
        - Load IDs from PROCESSED_DIR / "embedding_ids.json"
        - Use get_es_client(), delete/create index with INDEX_MAPPINGS
        - Use _generate_actions(corpus_path, embeddings, ids) for bulk data
        - Use elasticsearch.helpers.bulk() with chunk_size=100
        - Call es.indices.refresh() after bulk ingest
    """
    corpus_path = RAW_DIR / "corpus.jsonl"
    embeddings_path = PROCESSED_DIR / "embeddings.npy"
    ids_path = PROCESSED_DIR / "embedding_ids.json"
    for path in (corpus_path, embeddings_path, ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    ids = [str(doc_id) for doc_id in json.loads(ids_path.read_text(encoding="utf-8"))]
    if embeddings.shape != (len(ids), 4096):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match ({len(ids)}, 4096)."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("embedding_ids.json contains duplicate document IDs.")

    es = get_es_client()
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPINGS)

    count, _ = bulk(
        es,
        _generate_actions(corpus_path, embeddings, ids),
        chunk_size=100,
        max_retries=3,
        raise_on_error=True,
        stats_only=True,
    )
    if count != len(ids):
        raise ValueError(
            f"Indexed {count} documents, but {len(ids)} embeddings were expected. "
            "Check corpus and embedding IDs."
        )
    es.indices.refresh(index=INDEX_NAME)
    if progress_callback:
        progress_callback(count)
    return count


if __name__ == "__main__":
    ingest()
