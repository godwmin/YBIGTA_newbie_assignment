from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from app import llm
from ingest import embedding
from ingest.elastic import ingest as elastic_ingest
from ingest.hybrid import ingest as hybrid_ingest
from ingest.pinecone import ingest as pinecone_ingest
from retrievers.elastic import retriever as elastic_retriever
from retrievers.hybrid import retriever as hybrid_retriever
from retrievers.pinecone import retriever as pinecone_retriever


class LLMTests(unittest.TestCase):
    def test_generate_no_rag_uses_required_options(self):
        create = MagicMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  Grace Bedell  "))]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch.dict(os.environ, {"UPSTAGE_API_KEY": "test-key"}, clear=False), patch.object(
            llm, "OpenAI", return_value=client
        ) as openai:
            answer = llm.generate("Who suggested Lincoln grow a beard?")

        self.assertEqual(answer, "Grace Bedell")
        openai.assert_called_once_with(api_key="test-key", base_url=llm.BASE_URL)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["max_tokens"], 1024)
        self.assertIn("Who suggested Lincoln", kwargs["messages"][0]["content"])
        self.assertNotIn("Context:", kwargs["messages"][0]["content"])

    def test_generate_rag_includes_context(self):
        create = MagicMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.dict(os.environ, {"UPSTAGE_API_KEY": "test-key"}, clear=False), patch.object(
            llm, "OpenAI", return_value=client
        ):
            llm.generate("Question?", context="Retrieved passage")

        prompt = create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Retrieved passage", prompt)
        self.assertIn("based ONLY", prompt)


class EmbeddingTests(unittest.TestCase):
    def test_embed_passages_writes_aligned_float32_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "PROCESSED_DIR": root,
                "EMBEDDINGS_PATH": root / "embeddings.npy",
                "IDS_PATH": root / "embedding_ids.json",
                "PARTIAL_EMBEDDINGS_PATH": root / "embeddings.partial.npy",
                "PARTIAL_IDS_PATH": root / "embedding_ids.partial.json",
                "BATCH_SIZE": 2,
            }

            def fake_embed(_client, texts):
                return [[float(len(text))] * embedding.DIM for text in texts]

            progress = []
            with patch.multiple(embedding, **paths), patch.object(
                embedding, "_require_api_keys", return_value=["key-1", "key-2"]
            ), patch.object(embedding, "OpenAI", return_value=object()), patch.object(
                embedding, "_embed_batch_safe", side_effect=fake_embed
            ):
                result = embedding.embed_passages(
                    ["one", "three", "seven"],
                    ["a", "b", "c"],
                    progress_callback=lambda current, total: progress.append((current, total)),
                )

                self.assertEqual(result.shape, (3, embedding.DIM))
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result[:, 0].tolist(), [3.0, 5.0, 5.0])
                self.assertEqual(json.loads(paths["IDS_PATH"].read_text()), ["a", "b", "c"])
                self.assertTrue(paths["EMBEDDINGS_PATH"].exists())
                self.assertFalse(paths["PARTIAL_EMBEDDINGS_PATH"].exists())
                self.assertEqual(progress[-1], (2, 2))

                # An exact second call must use the completed cache, not the API.
                with patch.object(
                    embedding, "_embed_batch_safe", side_effect=AssertionError("API called")
                ):
                    cached = embedding.embed_passages(
                        ["one", "three", "seven"], ["a", "b", "c"]
                    )
                np.testing.assert_array_equal(cached, result)

    def test_embed_query_uses_query_model(self):
        create = MagicMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.25] * embedding.DIM)]
            )
        )
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        with patch.object(embedding, "_require_api_keys", return_value=["key"]), patch.object(
            embedding, "OpenAI", return_value=client
        ):
            result = embedding.embed_query("semantic question")

        self.assertEqual(len(result), embedding.DIM)
        self.assertEqual(
            create.call_args.kwargs["model"], "solar-embedding-1-large-query"
        )


class ElasticsearchTests(unittest.TestCase):
    def test_bm25_ingest_recreates_and_refreshes_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            corpus_path = raw_dir / "corpus.jsonl"
            corpus_path.write_text(
                '\n'.join(
                    json.dumps({"id": str(i), "text": f"document {i}"})
                    for i in range(2)
                ),
                encoding="utf-8",
            )
            indices = MagicMock()
            indices.exists.return_value = True
            es = SimpleNamespace(indices=indices)

            def fake_bulk(_es, actions, **_kwargs):
                materialized = list(actions)
                self.assertEqual(materialized[0]["_source"]["text"], "document 0")
                return len(materialized), 0

            progress = []
            with patch.object(elastic_ingest, "RAW_DIR", raw_dir), patch.object(
                elastic_ingest, "get_es_client", return_value=es
            ), patch.object(elastic_ingest, "bulk", side_effect=fake_bulk):
                count = elastic_ingest.ingest(progress_callback=progress.append)

            self.assertEqual(count, 2)
            indices.delete.assert_called_once_with(index=elastic_ingest.INDEX_NAME)
            indices.create.assert_called_once_with(
                index=elastic_ingest.INDEX_NAME,
                mappings=elastic_ingest.INDEX_MAPPINGS,
            )
            indices.refresh.assert_called_once_with(index=elastic_ingest.INDEX_NAME)
            self.assertEqual(progress, [2])

    def test_bm25_retriever_returns_contract(self):
        es = MagicMock()
        es.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "42", "_score": 3.5, "_source": {"text": "passage"}}
                ]
            }
        }
        with patch.object(elastic_retriever, "get_es_client", return_value=es):
            result = elastic_retriever.search("Lincoln", top_k=3)

        self.assertEqual(
            result,
            [{"id": "42", "text": "passage", "score": 3.5, "method": "BM25"}],
        )
        self.assertEqual(es.search.call_args.kwargs["size"], 3)

    def test_hybrid_retriever_builds_rrf_request(self):
        es = MagicMock()
        es.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "7", "_score": 0.03, "_source": {"text": "hybrid"}}
                ]
            }
        }
        with patch.object(
            hybrid_retriever, "embed_query", return_value=[0.0] * 4096
        ), patch.object(hybrid_retriever, "get_es_client", return_value=es):
            result = hybrid_retriever.search("query", top_k=5, candidate_size=20)

        request = es.search.call_args.kwargs
        rrf = request["retriever"]["rrf"]
        self.assertEqual(rrf["rank_constant"], 60)
        self.assertEqual(rrf["rank_window_size"], 20)
        self.assertEqual(rrf["retrievers"][1]["knn"]["num_candidates"], 40)
        self.assertEqual(result[0]["method"], "Hybrid (RRF)")


class PineconeTests(unittest.TestCase):
    def test_vector_retriever_queries_with_metadata(self):
        index = MagicMock()
        index.query.return_value = {
            "matches": [
                {"id": "9", "score": 0.91, "metadata": {"text": "vector hit"}}
            ]
        }
        pc = MagicMock()
        pc.Index.return_value = index
        with patch.dict(
            os.environ,
            {"PINECONE_API_KEY": "pc-key", "PINECONE_INDEX": "test-index"},
            clear=False,
        ), patch.object(
            pinecone_retriever, "embed_query", return_value=[0.0] * 4096
        ), patch.object(pinecone_retriever, "Pinecone", return_value=pc):
            result = pinecone_retriever.search("query", top_k=4)

        self.assertEqual(
            result,
            [{"id": "9", "text": "vector hit", "score": 0.91, "method": "Vector"}],
        )
        self.assertTrue(index.query.call_args.kwargs["include_metadata"])
        self.assertFalse(index.query.call_args.kwargs["include_values"])

    def test_pinecone_ingest_preserves_id_vector_text_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            raw_dir.mkdir()
            processed_dir.mkdir()
            np.save(
                processed_dir / "embeddings.npy",
                np.asarray([[1.0] * 4096, [2.0] * 4096], dtype=np.float32),
            )
            (processed_dir / "embedding_ids.json").write_text(
                json.dumps(["b", "a"]), encoding="utf-8"
            )
            (raw_dir / "corpus.jsonl").write_text(
                '\n'.join(
                    [
                        json.dumps({"id": "a", "text": "text A"}),
                        json.dumps({"id": "b", "text": "text B"}),
                    ]
                ),
                encoding="utf-8",
            )

            index = MagicMock()
            pc = MagicMock()
            pc.list_indexes.return_value.names.return_value = ["ragsession"]
            pc.describe_index.return_value = SimpleNamespace(
                dimension=4096, metric="cosine"
            )
            pc.Index.return_value = index

            with patch.dict(os.environ, {"PINECONE_API_KEY": "pc-key"}, clear=False), patch.object(
                pinecone_ingest, "RAW_DIR", raw_dir
            ), patch.object(
                pinecone_ingest, "PROCESSED_DIR", processed_dir
            ), patch.object(
                pinecone_ingest, "Pinecone", return_value=pc
            ):
                count = pinecone_ingest.ingest()

            self.assertEqual(count, 2)
            vectors = index.upsert.call_args.kwargs["vectors"]
            self.assertEqual(vectors[0]["id"], "b")
            self.assertEqual(vectors[0]["values"][0], 1.0)
            self.assertEqual(vectors[0]["metadata"]["text"], "text B")


class ValidationTests(unittest.TestCase):
    def test_searches_reject_empty_queries_before_network_calls(self):
        for search in (
            elastic_retriever.search,
            pinecone_retriever.search,
            hybrid_retriever.search,
        ):
            with self.subTest(search=search.__module__):
                with self.assertRaises(ValueError):
                    search("   ")


if __name__ == "__main__":
    unittest.main()
