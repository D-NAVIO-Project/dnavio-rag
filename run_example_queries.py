"""Run five example RAG queries against the PostgreSQL/pgvector index."""

from __future__ import annotations

import argparse
import json
import os
import sys

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXAMPLE_QUERIES = [
    ("Generator phase imbalance", "What causes generator phase imbalance?", []),
    ("Governor speed signal", "How should a failed governor speed signal be diagnosed?", [("failure_class", "Data/Input Failure")]),
    ("Boiler flame failure", "What are the causes and consequences of auxiliary boiler flame failure?", []),
    ("Oil purifier maintenance", "What maintenance controls reduce oil purifier failure risk?", []),
    ("High-severity failures", "What preventive controls are recommended for high-severity failure modes?", [("severity", "10")]),
]


def filters_from_pairs(pairs: list[tuple[str, str]]) -> dict | None:
    if not pairs:
        return None
    return {
        "operator": "AND",
        "conditions": [
            {"field": f"meta.{key}", "operator": "==", "value": value}
            for key, value in pairs
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-string", default=os.getenv("PG_CONN_STR", "postgresql://rag_user:rag_password@localhost:5432/rag_db"), help="PostgreSQL URL; defaults to PG_CONN_STR")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.connection_string:
        print("Set PG_CONN_STR or pass --connection-string.", file=sys.stderr)
        return 2

    document_store = PgvectorDocumentStore(
        connection_string=Secret.from_token(args.connection_string),
        table_name="rag_documents",
        embedding_dimension=384,
        create_extension=True,
        search_strategy="hnsw",
    )
    retriever = PgvectorEmbeddingRetriever(document_store=document_store)
    embedder = SentenceTransformersTextEmbedder(model=args.model)
    embedder.warm_up()

    for number, (title, question, filter_pairs) in enumerate(EXAMPLE_QUERIES, start=1):
        query_embedding = embedder.run(text=question)["embedding"]
        documents = retriever.run(
            query_embedding=query_embedding,
            top_k=args.top_k,
            filters=filters_from_pairs(filter_pairs),
        )["documents"]
        results = [
            {"score": document.score, "content": document.content, "meta": document.meta}
            for document in documents
        ]
        print(f"\n{'=' * 80}\n{number}. {title}\nQuestion: {question}\nResults:\n")
        print(json.dumps(results, indent=2, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())