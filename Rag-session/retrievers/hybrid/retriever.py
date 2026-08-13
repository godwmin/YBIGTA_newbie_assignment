"""Hybrid retriever using Elasticsearch RRF (Reciprocal Rank Fusion).

Combines BM25 text search with dense vector kNN search.
Uses ES 8.14+ RRF support with rank_constant=60.
"""

import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from ingest.embedding import embed_query

load_dotenv()

INDEX_NAME = "wiki-hybrid"


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
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


def search(query: str, top_k: int = 10, candidate_size: int = 50) -> list[dict]:
    """RRF hybrid search combining BM25 + kNN.

    Args:
        query: Search query string.
        top_k: Number of results to return.
        candidate_size: Number of kNN candidates before RRF fusion.

    Returns:
        list[dict], each dict has keys: "id", "text", "score", "method".
        "method" should be "Hybrid (RRF)".

    Hints:
        - Use embed_query(query) to get the query embedding vector
        - Use get_es_client() and es.search() with "retriever" parameter
        - RRF retriever combines "standard" (BM25 match) + "knn" retrievers
        - kNN field: "embedding", rank_constant: 60
        - num_candidates = candidate_size * 2
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if (
        not isinstance(candidate_size, int)
        or isinstance(candidate_size, bool)
        or candidate_size <= 0
    ):
        raise ValueError("candidate_size must be a positive integer.")

    candidate_size = max(candidate_size, top_k)
    vector = embed_query(query.strip())
    retriever = {
        "rrf": {
            "retrievers": [
                {
                    "standard": {
                        "query": {"match": {"text": {"query": query.strip()}}}
                    }
                },
                {
                    "knn": {
                        "field": "embedding",
                        "query_vector": vector,
                        "k": candidate_size,
                        "num_candidates": candidate_size * 2,
                    }
                },
            ],
            "rank_constant": 60,
            "rank_window_size": candidate_size,
        }
    }
    response = get_es_client().search(
        index=INDEX_NAME,
        retriever=retriever,
        size=top_k,
        source=["text"],
    )
    return [
        {
            "id": str(hit["_id"]),
            "text": str(hit.get("_source", {}).get("text", "")),
            "score": float(hit.get("_score") or 0.0),
            "method": "Hybrid (RRF)",
        }
        for hit in response.get("hits", {}).get("hits", [])
    ]
