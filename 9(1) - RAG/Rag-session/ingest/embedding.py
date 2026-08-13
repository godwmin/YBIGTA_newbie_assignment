"""Upstage Solar embedding utility with disk caching and parallel API keys.

Models:
  - solar-embedding-1-large-passage  (document encoding)
  - solar-embedding-1-large-query    (query encoding)

Uses multiple API keys (UPSTAGE_API_KEY1..N) for parallel embedding.
Each key gets its own thread with independent RPM/TPM limits.
Saves progress incrementally so crashes don't lose work.
Cache: data/processed/embeddings.npy (float32) + embedding_ids.json
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from threading import Lock

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
IDS_PATH = PROCESSED_DIR / "embedding_ids.json"

BATCH_SIZE = 100
RPM_LIMIT = 100
MIN_INTERVAL = 60.0 / RPM_LIMIT
DIM = 4096
BASE_URL = "https://api.upstage.ai/v1/solar"
MAX_CHARS = 12000  # ~3000 tokens, safely under 4000 token limit
MAX_RETRIES = 3
PARTIAL_EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.partial.npy"
PARTIAL_IDS_PATH = PROCESSED_DIR / "embedding_ids.partial.json"


def _get_api_keys() -> list[str]:
    """Collect all UPSTAGE_API_KEY* from env."""
    keys = []
    for i in range(1, 100):
        key = os.getenv(f"UPSTAGE_API_KEY{i}")
        if key:
            keys.append(key.strip())
        else:
            break
    if not keys:
        single = os.getenv("UPSTAGE_API_KEY", "")
        if single:
            keys.append(single.strip())
    # Do not run the same credential in multiple workers if it was configured
    # under more than one variable name.
    return list(dict.fromkeys(keys))


def _require_api_keys() -> list[str]:
    keys = _get_api_keys()
    if not keys:
        raise RuntimeError(
            "Upstage API key is missing. Set UPSTAGE_API_KEY in the .env file."
        )
    return keys


def _truncate(text: str) -> str:
    """Truncate text to stay within token limits."""
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS]
    return text


def _embed_batch_safe(client: OpenAI, batch: list[str]) -> list[list[float]]:
    """Embed a batch with retry and fallback to smaller sub-batches."""
    truncated = [_truncate(t) for t in batch]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model="solar-embedding-1-large-passage",
                input=truncated,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            err_msg = str(e)
            if "maximum context length" in err_msg or "4000 tokens" in err_msg:
                # Split batch in half and process separately
                mid = len(truncated) // 2
                if mid == 0:
                    # Single text too long, truncate more aggressively
                    truncated = [t[:MAX_CHARS // 2] for t in truncated]
                    continue
                left = _embed_batch_safe(client, truncated[:mid])
                time.sleep(MIN_INTERVAL)
                right = _embed_batch_safe(client, truncated[mid:])
                return left + right
            elif attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                raise


def embed_passages(texts: list[str], ids: list[str], progress_callback=None) -> np.ndarray:
    """Embed passages using parallel API keys.

    Args:
        texts: List of passage strings to embed.
        ids: List of document IDs (same length as texts).
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        np.ndarray of shape (N, 4096), dtype float32.

    Hints:
        - Use _get_api_keys() to get API keys, OpenAI(api_key=..., base_url=BASE_URL) to create clients
        - Use _embed_batch_safe(client, batch) to embed a batch of texts
        - Process texts in chunks of BATCH_SIZE
        - Save results to EMBEDDINGS_PATH (.npy) and IDS_PATH (.json)
    """
    if len(texts) != len(ids):
        raise ValueError("texts and ids must have the same length.")
    if not texts:
        raise ValueError("At least one passage is required.")

    normalized_ids = [str(doc_id) for doc_id in ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Document IDs must be unique.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every passage must be a non-empty string.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Return an exact, valid cache immediately. A cache with different IDs is
    # deliberately ignored because vector-to-document alignment is critical.
    try:
        cached = load_cached_embeddings()
    except (OSError, ValueError, json.JSONDecodeError):
        cached = None
    if cached is not None:
        cached_embeddings, cached_ids = cached
        if (
            cached_ids == normalized_ids
            and cached_embeddings.shape == (len(texts), DIM)
            and cached_embeddings.dtype == np.float32
            and np.isfinite(cached_embeddings).all()
        ):
            total = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            if progress_callback:
                progress_callback(total, total)
            return cached_embeddings

    keys = _require_api_keys()

    # A separate partial file makes interrupted jobs resumable without making
    # the Streamlit UI mistake an incomplete matrix for a finished result.
    can_resume = False
    if PARTIAL_EMBEDDINGS_PATH.exists() and PARTIAL_IDS_PATH.exists():
        try:
            partial_ids = json.loads(PARTIAL_IDS_PATH.read_text(encoding="utf-8"))
            partial = np.load(PARTIAL_EMBEDDINGS_PATH, mmap_mode="r+")
            can_resume = (
                partial_ids == normalized_ids
                and partial.shape == (len(texts), DIM)
                and partial.dtype == np.float32
            )
        except (OSError, ValueError, json.JSONDecodeError):
            can_resume = False

    if not can_resume:
        partial = np.lib.format.open_memmap(
            PARTIAL_EMBEDDINGS_PATH,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), DIM),
        )
        partial[:] = np.nan
        partial.flush()
        PARTIAL_IDS_PATH.write_text(
            json.dumps(normalized_ids, ensure_ascii=False), encoding="utf-8"
        )

    batches: list[list[int]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, len(texts))))
        missing = [idx for idx in indices if not np.isfinite(partial[idx]).all()]
        if missing:
            batches.append(missing)

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    completed_batches = total_batches - len(batches)
    if progress_callback and completed_batches:
        progress_callback(completed_batches, total_batches)

    lock = Lock()

    progress_events: Queue[int] = Queue()

    def worker(api_key: str, assigned_batches: list[list[int]]) -> int:
        nonlocal completed_batches
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        processed = 0
        for batch_number, indices in enumerate(assigned_batches):
            vectors = _embed_batch_safe(client, [texts[idx] for idx in indices])
            array = np.asarray(vectors, dtype=np.float32)
            if array.shape != (len(indices), DIM):
                raise ValueError(
                    f"Unexpected embedding shape {array.shape}; expected ({len(indices)}, {DIM})."
                )
            if not np.isfinite(array).all():
                raise ValueError("Embedding API returned NaN or infinite values.")

            with lock:
                partial[indices] = array
                partial.flush()
                completed_batches += 1
                current = completed_batches
            progress_events.put(current)
            processed += len(indices)

            if batch_number < len(assigned_batches) - 1:
                time.sleep(MIN_INTERVAL)
        return processed

    assignments = [batches[i::len(keys)] for i in range(len(keys))]
    with ThreadPoolExecutor(max_workers=len(keys)) as executor:
        futures = [
            executor.submit(worker, key, assigned)
            for key, assigned in zip(keys, assignments)
            if assigned
        ]
        reported = 0
        while reported < len(batches):
            try:
                current = progress_events.get(timeout=0.1)
                reported += 1
                if progress_callback:
                    progress_callback(current, total_batches)
            except Empty:
                if all(future.done() for future in futures):
                    break
        for future in as_completed(futures):
            future.result()

    if not np.isfinite(partial).all():
        raise RuntimeError("Embedding job finished with incomplete rows.")

    partial.flush()
    del partial
    os.replace(PARTIAL_EMBEDDINGS_PATH, EMBEDDINGS_PATH)
    IDS_PATH.write_text(
        json.dumps(normalized_ids, ensure_ascii=False), encoding="utf-8"
    )
    PARTIAL_IDS_PATH.unlink(missing_ok=True)
    return np.load(EMBEDDINGS_PATH)


def embed_query(query: str) -> list[float]:
    """Embed a single query using the query model.

    Args:
        query: The search query string.

    Returns:
        list[float] of length 4096 (embedding vector).

    Hints:
        - Use _get_api_keys() to get an API key
        - Model name: "solar-embedding-1-large-query"
        - Use _truncate() to handle long queries
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    key = _require_api_keys()[0]
    client = OpenAI(api_key=key, base_url=BASE_URL)
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model="solar-embedding-1-large-query",
                input=_truncate(query.strip()),
            )
            if not response.data:
                raise ValueError("Embedding API returned no data.")
            vector = list(response.data[0].embedding)
            if len(vector) != DIM:
                raise ValueError(
                    f"Unexpected query embedding dimension {len(vector)}; expected {DIM}."
                )
            if not np.isfinite(np.asarray(vector, dtype=np.float32)).all():
                raise ValueError("Embedding API returned NaN or infinite values.")
            return vector
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))

    raise RuntimeError("Failed to embed the query after retries.") from last_error


def load_cached_embeddings() -> tuple[np.ndarray, list[str]] | None:
    """Load cached embeddings from disk. Returns (embeddings, ids) or None."""
    if EMBEDDINGS_PATH.exists() and IDS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        return embeddings, ids
    return None


if __name__ == "__main__":
    from data.download import RAW_DIR

    corpus_path = RAW_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        print("Run data/download.py first.")
        raise SystemExit(1)

    texts, ids = [], []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["id"])
            texts.append(doc["text"])

    embed_passages(texts, ids)
