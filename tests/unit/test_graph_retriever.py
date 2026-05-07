"""Unit tests for BigQueryGraphVectorContextRetriever and BigQueryGraphTextToGQLRetriever."""

from unittest.mock import MagicMock, create_autospec

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from langchain_bigquery.graph_retriever import (
    BigQueryGraphTextToGQLRetriever,
    BigQueryGraphVectorContextRetriever,
    DistanceStrategy,
    _convert_to_doc,
    _get_graph_name_from_schema,
)
from langchain_bigquery.graph_store import BigQueryGraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_store(graph_name: str = "test_graph") -> BigQueryGraphStore:
    client = MagicMock()
    job = MagicMock()
    job.result.return_value = iter([])
    client.query.return_value = job

    store = BigQueryGraphStore(
        project_id="proj",
        dataset_id="ds",
        graph_name=graph_name,
        client=client,
    )
    return store


def _make_mock_store_with_schema(
    graph_name: str = "test_graph", label: str = "Person"
) -> BigQueryGraphStore:
    store = _make_mock_store(graph_name)
    from langchain_bigquery.graph_store import ElementSchema
    from requests.structures import CaseInsensitiveDict

    node = ElementSchema.make_node_schema(
        label,
        label,
        graph_name,
        "ds",
        CaseInsensitiveDict({"id": "STRING", "name": "STRING", "embedding": "ARRAY<FLOAT64>"}),
    )
    store.schema.nodes[label] = node
    store.schema.node_tables[node.base_table_name] = node
    store.schema._update_labels_and_properties(node)
    return store


def _make_mock_embeddings(dim: int = 3):
    emb = create_autospec(Embeddings, instance=True)
    emb.embed_query.return_value = [0.1] * dim
    emb.embed_documents.return_value = [[0.1] * dim]
    return emb


def _make_mock_llm():
    llm = create_autospec(BaseLanguageModel, instance=True)
    return llm


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_convert_to_doc(self):
        doc = _convert_to_doc({"name": "Alice", "age": 30})
        assert "Alice" in doc.page_content

    def test_get_graph_name_from_schema(self):
        import json

        schema = json.dumps({"Name of graph": "my_graph"})
        assert _get_graph_name_from_schema(schema) == "`my_graph`"


# ---------------------------------------------------------------------------
# BigQueryGraphVectorContextRetriever tests
# ---------------------------------------------------------------------------


class TestBigQueryGraphVectorContextRetriever:
    def test_init_requires_embedding_service(self):
        store = _make_mock_store()
        with pytest.raises(ValueError, match="embedding_service"):
            BigQueryGraphVectorContextRetriever(
                graph_store=store,
                expand_by_hops=1,
            )

    def test_init_requires_exactly_one_option(self):
        store = _make_mock_store()
        emb = _make_mock_embeddings()
        with pytest.raises(ValueError, match="One and only one"):
            BigQueryGraphVectorContextRetriever(
                graph_store=store,
                embedding_service=emb,
            )

    def test_init_with_expand_by_hops(self):
        store = _make_mock_store()
        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever(
            graph_store=store,
            embedding_service=emb,
            expand_by_hops=2,
        )
        assert retriever.expand_by_hops == 2

    def test_init_with_return_properties(self):
        store = _make_mock_store()
        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever(
            graph_store=store,
            embedding_service=emb,
            return_properties_list=["name", "age"],
        )
        assert retriever.return_properties_list == ["name", "age"]

    def test_from_params(self):
        store = _make_mock_store()
        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever.from_params(
            embedding_service=emb,
            graph_store=store,
            expand_by_hops=0,
        )
        assert retriever.embedding_service is emb

    def test_distance_strategy_default(self):
        store = _make_mock_store()
        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever(
            graph_store=store,
            embedding_service=emb,
            expand_by_hops=1,
        )
        assert retriever.distance_strategy == DistanceStrategy.COSINE

    def test_get_relevant_documents_return_properties(self):
        store = _make_mock_store_with_schema(label="Person")
        store.query = MagicMock(return_value=[{"name": "Alice"}, {"name": "Bob"}])

        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever(
            graph_store=store,
            embedding_service=emb,
            label_expr="Person",
            return_properties_list=["name"],
        )
        docs = retriever.invoke("Who is Alice?")
        assert len(docs) == 2
        store.query.assert_called_once()
        query_str = store.query.call_args[0][0]
        assert "COSINE_DISTANCE" in query_str
        assert "test_graph_Person" in query_str

    def test_get_relevant_documents_expand_hops_0(self):
        store = _make_mock_store_with_schema(label="Person")
        store.query = MagicMock(
            side_effect=[
                [{"id": "1"}],
                [{"path": '{"id": "1"}'}],
            ]
        )

        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever(
            graph_store=store,
            embedding_service=emb,
            label_expr="Person",
            expand_by_hops=0,
        )
        docs = retriever.invoke("test query")
        assert len(docs) == 1
        assert store.query.call_count == 2
        sql_query = store.query.call_args_list[0][0][0]
        assert "COSINE_DISTANCE" in sql_query
        gql_query = store.query.call_args_list[1][0][0]
        assert "TO_JSON" in gql_query
        assert "GRAPH" in gql_query

    def test_euclidean_distance(self):
        store = _make_mock_store_with_schema(label="Person")
        store.query = MagicMock(return_value=[{"name": "X"}])

        emb = _make_mock_embeddings()
        retriever = BigQueryGraphVectorContextRetriever(
            graph_store=store,
            embedding_service=emb,
            label_expr="Person",
            return_properties_list=["name"],
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        docs = retriever.invoke("test")
        query_str = store.query.call_args[0][0]
        assert "EUCLIDEAN_DISTANCE" in query_str


# ---------------------------------------------------------------------------
# BigQueryGraphTextToGQLRetriever tests
# ---------------------------------------------------------------------------


class TestBigQueryGraphTextToGQLRetriever:
    def test_from_params_requires_llm(self):
        store = _make_mock_store()
        with pytest.raises(ValueError, match="llm"):
            BigQueryGraphTextToGQLRetriever.from_params(llm=None, graph_store=store)

    def test_from_params_with_llm(self):
        store = _make_mock_store()
        llm = _make_mock_llm()
        retriever = BigQueryGraphTextToGQLRetriever.from_params(
            llm=llm, graph_store=store
        )
        assert retriever.llm is llm
        assert retriever.selector is None

    def test_add_example_requires_selector(self):
        store = _make_mock_store()
        llm = _make_mock_llm()
        retriever = BigQueryGraphTextToGQLRetriever(graph_store=store, llm=llm)
        with pytest.raises(ValueError, match="selector"):
            retriever.add_example("question", "GRAPH g MATCH (n) RETURN n")
