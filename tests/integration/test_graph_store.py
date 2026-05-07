"""Integration tests for BigQueryGraphStore and Graph Retrievers.

These tests require a live BigQuery connection and GCP credentials.
Run with: pytest tests/integration/ -m integration

Environment variables required:
  GOOGLE_CLOUD_PROJECT  - GCP project ID
  BIGQUERY_DATASET      - BigQuery dataset for test graphs
"""

from __future__ import annotations

import os
import uuid
from typing import Generator

import pytest
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from langchain_bigquery import (
    BigQueryGraphStore,
    BigQueryGraphTextToGQLRetriever,
    BigQueryGraphVectorContextRetriever,
    DistanceStrategy,
)

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "test_graph_rag")

SUFFIX = uuid.uuid4().hex[:8]


def _make_graph_name() -> str:
    return f"test_graph_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

PEOPLE = [
    Node(id="alice", type="Person", properties={"name": "Alice", "age": 30}),
    Node(id="bob", type="Person", properties={"name": "Bob", "age": 25}),
    Node(id="carol", type="Person", properties={"name": "Carol", "age": 35}),
]

CITIES = [
    Node(id="seoul", type="City", properties={"name": "Seoul", "country": "Korea"}),
    Node(
        id="tokyo", type="City", properties={"name": "Tokyo", "country": "Japan"}
    ),
]

RELATIONSHIPS = [
    Relationship(
        source=PEOPLE[0],
        target=PEOPLE[1],
        type="KNOWS",
        properties={"since": 2020},
    ),
    Relationship(
        source=PEOPLE[1],
        target=PEOPLE[2],
        type="KNOWS",
        properties={"since": 2021},
    ),
    Relationship(
        source=PEOPLE[0],
        target=CITIES[0],
        type="LIVES_IN",
        properties={},
    ),
    Relationship(
        source=PEOPLE[1],
        target=CITIES[1],
        type="LIVES_IN",
        properties={},
    ),
    Relationship(
        source=PEOPLE[2],
        target=CITIES[0],
        type="LIVES_IN",
        properties={},
    ),
]

GRAPH_DOC = GraphDocument(
    nodes=PEOPLE + CITIES,
    relationships=RELATIONSHIPS,
    source=Document(
        page_content="Sample people and cities graph.",
        metadata={"source": "test"},
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_store() -> Generator[BigQueryGraphStore, None, None]:
    if not GOOGLE_CLOUD_PROJECT:
        pytest.skip("GOOGLE_CLOUD_PROJECT not set")

    graph_name = _make_graph_name()
    store = BigQueryGraphStore(
        project_id=GOOGLE_CLOUD_PROJECT,
        dataset_id=BIGQUERY_DATASET,
        graph_name=graph_name,
        location=GOOGLE_CLOUD_LOCATION,
    )
    store.add_graph_documents([GRAPH_DOC])
    store.refresh_schema()

    yield store

    store.cleanup()


@pytest.fixture(scope="module")
def embedding_model():
    try:
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(
            model_name=os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-004"),
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
    except ImportError:
        pytest.skip("langchain-google-vertexai not installed")


@pytest.fixture(scope="module")
def llm():
    try:
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model_name="gemini-2.0-flash",
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
    except ImportError:
        pytest.skip("langchain-google-vertexai not installed")


# ---------------------------------------------------------------------------
# GraphStore tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBigQueryGraphStore:
    def test_add_and_query_nodes(self, graph_store) -> None:
        """Nodes inserted via add_graph_documents should be queryable."""
        graph_name = graph_store.schema.graph_name
        results = graph_store.query(
            f"GRAPH `{BIGQUERY_DATASET}`.`{graph_name}` "
            "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name"
        )
        names = [r["name"] for r in results]
        assert "Alice" in names
        assert "Bob" in names
        assert "Carol" in names

    def test_add_and_query_edges(self, graph_store) -> None:
        """Edges inserted via add_graph_documents should be queryable."""
        graph_name = graph_store.schema.graph_name
        results = graph_store.query(
            f"GRAPH `{BIGQUERY_DATASET}`.`{graph_name}` "
            "MATCH (a:Person)-[:KNOWS]->(b:Person) "
            "RETURN a.name AS src, b.name AS dst ORDER BY a.name"
        )
        assert len(results) >= 2
        pairs = [(r["src"], r["dst"]) for r in results]
        assert ("Alice", "Bob") in pairs

    def test_get_schema_not_empty(self, graph_store) -> None:
        """Schema string should contain node/edge type info."""
        schema = graph_store.get_schema
        assert "Person" in schema
        assert "City" in schema

    def test_get_structured_schema(self, graph_store) -> None:
        """Structured schema should be a valid dict."""
        structured = graph_store.get_structured_schema
        assert isinstance(structured, dict)

    def test_get_ddl(self, graph_store) -> None:
        """DDL string should reference the graph name."""
        ddl = graph_store.get_ddl()
        assert graph_store.schema.graph_name in ddl

    def test_query_with_params(self, graph_store) -> None:
        """Parameterized queries should work."""
        graph_name = graph_store.schema.graph_name
        results = graph_store.query(
            f"GRAPH `{BIGQUERY_DATASET}`.`{graph_name}` "
            "MATCH (p:Person) WHERE p.name = @name RETURN p.name AS name",
            params={"name": "Alice"},
        )
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_add_duplicate_nodes_merges(self, graph_store) -> None:
        """Adding the same node again should merge properties, not duplicate."""
        graph_name = graph_store.schema.graph_name

        updated_node = Node(
            id="alice", type="Person", properties={"name": "Alice", "age": 31}
        )
        doc = GraphDocument(
            nodes=[updated_node],
            relationships=[],
            source=Document(page_content="update", metadata={}),
        )
        graph_store.add_graph_documents([doc])

        results = graph_store.query(
            f"GRAPH `{BIGQUERY_DATASET}`.`{graph_name}` "
            "MATCH (p:Person {{id: 'alice'}}) RETURN p.age AS age"
        )
        assert len(results) == 1
        assert results[0]["age"] == 31

    def test_graph_traversal(self, graph_store) -> None:
        """Multi-hop traversal should work."""
        graph_name = graph_store.schema.graph_name
        results = graph_store.query(
            f"GRAPH `{BIGQUERY_DATASET}`.`{graph_name}` "
            "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:LIVES_IN]->(c:City) "
            "RETURN a.name AS person, c.name AS city ORDER BY a.name"
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# GraphStore lifecycle test (create + cleanup)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBigQueryGraphStoreLifecycle:
    def test_create_and_cleanup(self) -> None:
        """A fresh graph can be created, populated, and cleaned up."""
        if not GOOGLE_CLOUD_PROJECT:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        graph_name = _make_graph_name()
        store = BigQueryGraphStore(
            project_id=GOOGLE_CLOUD_PROJECT,
            dataset_id=BIGQUERY_DATASET,
            graph_name=graph_name,
            location=GOOGLE_CLOUD_LOCATION,
        )

        simple_doc = GraphDocument(
            nodes=[
                Node(id="x", type="TestNode", properties={"val": 1}),
                Node(id="y", type="TestNode", properties={"val": 2}),
            ],
            relationships=[
                Relationship(
                    source=Node(id="x", type="TestNode"),
                    target=Node(id="y", type="TestNode"),
                    type="TestEdge",
                    properties={},
                )
            ],
            source=Document(page_content="lifecycle", metadata={}),
        )
        store.add_graph_documents([simple_doc])
        store.refresh_schema()

        assert "TestNode" in store.get_schema

        store.cleanup()

    def test_flexible_schema(self) -> None:
        """Flexible schema mode should handle dynamic properties."""
        if not GOOGLE_CLOUD_PROJECT:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        graph_name = _make_graph_name()
        store = BigQueryGraphStore(
            project_id=GOOGLE_CLOUD_PROJECT,
            dataset_id=BIGQUERY_DATASET,
            graph_name=graph_name,
            location=GOOGLE_CLOUD_LOCATION,
            use_flexible_schema=True,
        )

        try:
            doc1 = GraphDocument(
                nodes=[
                    Node(id="a", type="Flex", properties={"color": "red"}),
                    Node(id="b", type="Flex", properties={"size": 10}),
                ],
                relationships=[],
                source=Document(page_content="flex", metadata={}),
            )
            store.add_graph_documents([doc1])

            doc2 = GraphDocument(
                nodes=[
                    Node(
                        id="c",
                        type="Flex",
                        properties={"color": "blue", "weight": 5.0},
                    ),
                ],
                relationships=[],
                source=Document(page_content="flex2", metadata={}),
            )
            store.add_graph_documents([doc2])
            store.refresh_schema()

            assert "Flex" in store.get_schema
        finally:
            store.cleanup()


# ---------------------------------------------------------------------------
# TextToGQL Retriever tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBigQueryGraphTextToGQLRetriever:
    def test_retrieve_with_llm(self, graph_store, llm) -> None:
        """TextToGQL retriever should return documents from a natural language query."""
        retriever = BigQueryGraphTextToGQLRetriever.from_params(
            graph_store=graph_store,
            llm=llm,
        )
        results = retriever.invoke("Who does Alice know?")
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)

    def test_retrieve_with_few_shot(self, graph_store, llm, embedding_model) -> None:
        """TextToGQL with few-shot examples should produce results."""
        graph_name = graph_store.schema.graph_name

        retriever = BigQueryGraphTextToGQLRetriever.from_params(
            graph_store=graph_store,
            llm=llm,
            embedding_service=embedding_model,
        )
        retriever.add_example(
            question="Who does Alice know?",
            gql=(
                f"GRAPH `{BIGQUERY_DATASET}`.`{graph_name}` "
                "MATCH (a:Person {id: 'alice'})-[:KNOWS]->(b:Person) "
                "RETURN b.name AS name"
            ),
        )
        results = retriever.invoke("Who does Bob know?")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# VectorContext Retriever tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBigQueryGraphVectorContextRetriever:
    @pytest.fixture(scope="class")
    def vector_graph_store(self, embedding_model) -> Generator:
        """Graph store with embedding-bearing nodes."""
        if not GOOGLE_CLOUD_PROJECT:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        graph_name = _make_graph_name()
        store = BigQueryGraphStore(
            project_id=GOOGLE_CLOUD_PROJECT,
            dataset_id=BIGQUERY_DATASET,
            graph_name=graph_name,
            location=GOOGLE_CLOUD_LOCATION,
        )

        embeddings = embedding_model.embed_documents(
            ["Alice is a software engineer", "Bob is a data scientist"]
        )

        nodes = [
            Node(
                id="alice",
                type="Person",
                properties={
                    "name": "Alice",
                    "bio": "Alice is a software engineer",
                    "embedding": embeddings[0],
                },
            ),
            Node(
                id="bob",
                type="Person",
                properties={
                    "name": "Bob",
                    "bio": "Bob is a data scientist",
                    "embedding": embeddings[1],
                },
            ),
        ]
        doc = GraphDocument(
            nodes=nodes,
            relationships=[
                Relationship(
                    source=nodes[0],
                    target=nodes[1],
                    type="COLLABORATES",
                    properties={},
                )
            ],
            source=Document(page_content="vector test", metadata={}),
        )
        store.add_graph_documents([doc])
        store.refresh_schema()

        yield store

        store.cleanup()

    def test_return_properties(self, vector_graph_store, embedding_model) -> None:
        """VectorContext retriever with return_properties_list should return docs."""
        retriever = BigQueryGraphVectorContextRetriever.from_params(
            graph_store=vector_graph_store,
            embedding_service=embedding_model,
            label_expr="Person",
            embeddings_column="embedding",
            return_properties_list=["name", "bio"],
            top_k=2,
        )
        results = retriever.invoke("software engineer")
        assert len(results) > 0
        assert any("Alice" in doc.page_content for doc in results)

    def test_expand_by_hops_zero(self, vector_graph_store, embedding_model) -> None:
        """expand_by_hops=0 should return matched nodes only."""
        retriever = BigQueryGraphVectorContextRetriever.from_params(
            graph_store=vector_graph_store,
            embedding_service=embedding_model,
            label_expr="Person",
            embeddings_column="embedding",
            expand_by_hops=0,
            top_k=1,
        )
        results = retriever.invoke("data scientist")
        assert len(results) > 0

    def test_expand_by_hops_one(self, vector_graph_store, embedding_model) -> None:
        """expand_by_hops=1 should return neighbors as well."""
        retriever = BigQueryGraphVectorContextRetriever.from_params(
            graph_store=vector_graph_store,
            embedding_service=embedding_model,
            label_expr="Person",
            embeddings_column="embedding",
            expand_by_hops=1,
            top_k=1,
        )
        results = retriever.invoke("software engineer")
        assert len(results) > 0

    def test_euclidean_distance(self, vector_graph_store, embedding_model) -> None:
        """EUCLIDEAN distance strategy should also return results."""
        retriever = BigQueryGraphVectorContextRetriever.from_params(
            graph_store=vector_graph_store,
            embedding_service=embedding_model,
            label_expr="Person",
            embeddings_column="embedding",
            return_properties_list=["name"],
            top_k=2,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        results = retriever.invoke("engineer")
        assert len(results) > 0
