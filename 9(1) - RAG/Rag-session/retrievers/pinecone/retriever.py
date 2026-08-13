"""Vector retriever using Pinecone (cosine similarity)."""

import os

from dotenv import load_dotenv
from pinecone import Pinecone

from ingest.embedding import embed_query

load_dotenv()


def search(query: str, top_k: int = 10) -> list[dict]:
    """Vector cosine similarity search.

    Args:
        query: Search query string.
        top_k: Number of results to return.

    Returns:
        list[dict], each dict has keys: "id", "text", "score", "method".
        "method" should be "Vector".

    Hints:
        - Use embed_query(query) to get the query embedding vector
        - Connect: Pinecone(api_key=...) → pc.Index(index_name)
        - Use index.query(vector=..., top_k=..., include_metadata=True)
        - Text is in match["metadata"]["text"]
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    api_key = (os.getenv("PINECONE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Set it in the .env file.")
    index_name = (os.getenv("PINECONE_INDEX") or "ragsession").strip()

    vector = embed_query(query.strip())
    index = Pinecone(api_key=api_key).Index(index_name)
    response = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    matches = response.get("matches", []) if isinstance(response, dict) else response.matches

    results = []
    for match in matches:
        if isinstance(match, dict):
            match_id = match.get("id", "")
            score = match.get("score", 0.0)
            metadata = match.get("metadata") or {}
        else:
            match_id = match.id
            score = match.score
            metadata = match.metadata or {}
        results.append(
            {
                "id": str(match_id),
                "text": str(metadata.get("text", "")),
                "score": float(score or 0.0),
                "method": "Vector",
            }
        )
    return results
