"""BM25 retriever using Elasticsearch."""

import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

INDEX_NAME = "wiki-bm25"


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


def search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 match search.

    Args:
        query: Search query string.
        top_k: Number of results to return.

    Returns:
        list[dict], each dict has keys: "id", "text", "score", "method".
        "method" should be "BM25".

    Hints:
        - Use get_es_client() and es.search()
        - Index name: INDEX_NAME
        - Use "match" query on "text" field
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    response = get_es_client().search(
        index=INDEX_NAME,
        query={"match": {"text": {"query": query.strip()}}},
        size=top_k,
        source=["text"],
    )
    return [
        {
            "id": str(hit["_id"]),
            "text": str(hit.get("_source", {}).get("text", "")),
            "score": float(hit.get("_score") or 0.0),
            "method": "BM25",
        }
        for hit in response.get("hits", {}).get("hits", [])
    ]
