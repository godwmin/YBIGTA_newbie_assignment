"""Compare cosine, dot-product, and L2 retrieval on cached embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingest.embedding import embed_query

CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
IDS_PATH = PROJECT_ROOT / "data" / "processed" / "embedding_ids.json"


def _load_data() -> tuple[np.ndarray, list[str], dict[str, str]]:
    for path in (CORPUS_PATH, EMBEDDINGS_PATH, IDS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    ids = [str(value) for value in json.loads(IDS_PATH.read_text(encoding="utf-8"))]
    if embeddings.shape != (len(ids), 4096):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match ({len(ids)}, 4096)."
        )

    texts = {}
    with open(CORPUS_PATH, encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            texts[str(document["id"])] = str(document["text"])
    return embeddings, ids, texts


def compare(query: str, top_k: int) -> dict[str, list[dict]]:
    if not query.strip():
        raise ValueError("query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    embeddings, ids, texts = _load_data()
    query_vector = np.asarray(embed_query(query), dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)

    epsilon = np.finfo(np.float32).eps
    document_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)

    scores = {
        "Cosine similarity": (matrix @ query_vector)
        / np.maximum(document_norms * query_norm, epsilon),
        "Dot product": matrix @ query_vector,
        # Negate distance so all metrics can consistently be sorted descending.
        "L2 distance": -np.linalg.norm(matrix - query_vector, axis=1),
    }

    results = {}
    for method, values in scores.items():
        top_indices = np.argsort(values)[::-1][:top_k]
        results[method] = [
            {
                "rank": rank,
                "id": ids[index],
                "score": float(values[index]),
                "text": texts.get(ids[index], ""),
            }
            for rank, index in enumerate(top_indices, 1)
        ]
    return results


def render_markdown(query: str, results: dict[str, list[dict]]) -> str:
    lines = [f"## Query: {query}", ""]
    for method, rows in results.items():
        lines.extend(
            [
                f"### {method}",
                "",
                "| Rank | ID | Score | Passage preview |",
                "|---:|---|---:|---|",
            ]
        )
        for row in rows:
            preview = " ".join(row["text"].split())[:120].replace("|", "\\|")
            display_score = -row["score"] if method == "L2 distance" else row["score"]
            lines.append(
                f'| {row["rank"]} | {row["id"]} | {display_score:.6f} | {preview} |'
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to embed and compare")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()

    markdown = render_markdown(args.query, compare(args.query, args.top_k))
    if args.output:
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Saved results to {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
