"""Unit tests for BigQueryGraphStore (mocked BigQuery client)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from langchain_bigquery.graph_store import (
    BigQueryGraphSchema,
    BigQueryGraphStore,
    ElementSchema,
    _GraphDocUtil,
    partition_graph_docs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client():
    client = MagicMock()
    job = MagicMock()
    job.result.return_value = iter([])
    client.query.return_value = job
    return client


# ---------------------------------------------------------------------------
# partition_graph_docs tests
# ---------------------------------------------------------------------------


class TestPartitionGraphDocs:
    def test_empty(self):
        nodes, edges = partition_graph_docs([])
        assert nodes == {}
        assert edges == {}

    def test_single_doc(self):
        n1 = Node(id="1", type="Person", properties={"name": "Alice"})
        n2 = Node(id="2", type="Person", properties={"name": "Bob"})
        r = Relationship(source=n1, target=n2, type="KNOWS")
        doc = GraphDocument(nodes=[n1, n2], relationships=[r], source=Document(page_content=""))
        nodes, edges = partition_graph_docs([doc])
        assert "Person" in nodes
        assert len(nodes["Person"]) == 2
        assert len(edges) == 1

    def test_dedup_nodes(self):
        n1a = Node(id="1", type="Person", properties={"name": "Alice"})
        n1b = Node(id="1", type="Person", properties={"age": 30})
        doc = GraphDocument(nodes=[n1a, n1b], relationships=[], source=Document(page_content=""))
        nodes, edges = partition_graph_docs([doc])
        assert len(nodes["Person"]) == 1
        merged = nodes["Person"][0]
        assert merged.properties["name"] == "Alice"
        assert merged.properties["age"] == 30


# ---------------------------------------------------------------------------
# _GraphDocUtil tests
# ---------------------------------------------------------------------------


class TestGraphDocUtil:
    def test_valid_identifier(self):
        assert _GraphDocUtil.is_valid_identifier("my_table") is True
        assert _GraphDocUtil.is_valid_identifier("1bad") is False
        assert _GraphDocUtil.is_valid_identifier("") is False

    def test_fixup_identifier(self):
        assert _GraphDocUtil.fixup_identifier("Hello World!") == "Hello_World_"

    def test_to_identifier(self):
        assert _GraphDocUtil.to_identifier("foo") == "`foo`"


# ---------------------------------------------------------------------------
# BigQueryGraphSchema tests
# ---------------------------------------------------------------------------


class TestBigQueryGraphSchema:
    def test_init_valid(self):
        schema = BigQueryGraphSchema("test_graph", "my_dataset")
        assert schema.graph_name == "test_graph"
        assert schema.dataset_id == "my_dataset"

    def test_init_invalid_name(self):
        with pytest.raises(ValueError):
            BigQueryGraphSchema("invalid graph!", "ds")

    def test_repr_empty(self):
        schema = BigQueryGraphSchema("test_graph", "my_dataset")
        result = repr(schema)
        parsed = __import__("json").loads(result)
        assert parsed["Name of graph"] == "my_dataset.test_graph"

    def test_evolve_creates_ddls(self):
        schema = BigQueryGraphSchema("test_graph", "my_dataset")
        n1 = Node(id="1", type="Person", properties={"name": "Alice"})
        n2 = Node(id="2", type="Company", properties={"title": "Acme"})
        r = Relationship(source=n1, target=n2, type="WORKS_AT")
        doc = GraphDocument(nodes=[n1, n2], relationships=[r], source=Document(page_content=""))

        _GraphDocUtil.fixup_graph_documents([doc])
        ddls = schema.evolve([doc])

        assert len(ddls) > 0
        create_stmts = [d for d in ddls if d.startswith("CREATE TABLE")]
        assert len(create_stmts) >= 2
        graph_stmt = [d for d in ddls if "PROPERTY GRAPH" in d]
        assert len(graph_stmt) == 1

        graph_ddl = graph_stmt[0]
        assert "LABEL `WORKS_AT`" in graph_ddl
        assert "test_graph_Person_WORKS_AT_Company" in graph_ddl

    def test_evolve_excludes_array_from_graph_ddl(self):
        schema = BigQueryGraphSchema("test_graph", "my_dataset")
        embedding = [0.1, 0.2, 0.3]
        n1 = Node(id="1", type="Person", properties={"name": "Alice", "embedding": embedding})
        doc = GraphDocument(nodes=[n1], relationships=[], source=Document(page_content=""))

        _GraphDocUtil.fixup_graph_documents([doc])
        ddls = schema.evolve([doc])

        table_stmt = [d for d in ddls if d.startswith("CREATE TABLE")]
        assert any("embedding" in s for s in table_stmt)

        alter_stmts = [d for d in ddls if d.startswith("ALTER TABLE")]
        assert any("embedding" in s and "IF NOT EXISTS" in s for s in alter_stmts)

        graph_stmt = [d for d in ddls if "PROPERTY GRAPH" in d][0]
        assert "embedding" not in graph_stmt


# ---------------------------------------------------------------------------
# BigQueryGraphStore tests
# ---------------------------------------------------------------------------


class TestElementSchemaFromInfoSchema:
    def _make_node_info(self):
        return {
            "name": "Person",
            "key_columns": ["id"],
            "base_table_name": "test_graph_Person",
            "label_names": ["Person"],
            "property_definitions": [
                {
                    "property_declaration_name": "name",
                    "value_expression_sql": "name",
                },
            ],
        }

    def _make_edge_info(self):
        return {
            "name": "Person_KNOWS_Person",
            "key_columns": ["id", "target_id"],
            "base_table_name": "test_graph_Person_KNOWS_Person",
            "label_names": ["KNOWS"],
            "property_definitions": [],
            "source_node_table": {
                "node_table_name": "Person",
                "node_table_columns": ["id"],
                "edge_table_columns": ["id"],
            },
            "destination_node_table": {
                "node_table_name": "Person",
                "node_table_columns": ["id"],
                "edge_table_columns": ["target_id"],
            },
        }

    def _make_decl_by_types(self):
        from requests.structures import CaseInsensitiveDict

        return CaseInsensitiveDict({"name": "STRING"})

    def test_node_schema(self):
        schema = ElementSchema.from_info_schema(
            self._make_node_info(), self._make_decl_by_types(), kind="NODE"
        )
        assert schema.name == "Person"
        assert schema.kind == "NODE"
        assert schema.base_table_name == "test_graph_Person"
        assert schema.key_columns == ["id"]
        assert schema.labels == ["Person"]
        assert "name" in schema.properties

    def test_edge_schema(self):
        schema = ElementSchema.from_info_schema(
            self._make_edge_info(), self._make_decl_by_types(), kind="EDGE"
        )
        assert schema.name == "Person_KNOWS_Person"
        assert schema.kind == "EDGE"
        assert schema.base_table_name == "test_graph_Person_KNOWS_Person"
        assert schema.labels == ["KNOWS"]
        assert schema.source.node_name == "Person"
        assert schema.target.node_name == "Person"


class TestFromInformationSchema:
    def test_parses_bigquery_metadata(self):
        metadata = {
            "property_declarations": [
                {"name": "id", "type": "STRING"},
                {"name": "name", "type": "STRING"},
            ],
            "node_tables": [
                {
                    "name": "Person",
                    "key_columns": ["id"],
                    "base_table_name": "g_Person",
                    "label_names": ["Person"],
                    "property_definitions": [
                        {"property_declaration_name": "id", "value_expression_sql": "id"},
                        {"property_declaration_name": "name", "value_expression_sql": "name"},
                    ],
                },
            ],
            "edge_tables": [],
        }
        schema = BigQueryGraphSchema("test_graph", "my_dataset")
        schema.from_information_schema(metadata)
        assert "Person" in schema.nodes
        assert schema.nodes["Person"].base_table_name == "g_Person"


class TestBigQueryGraphStore:
    def test_init_with_mock_client(self):
        client = _make_mock_client()
        store = BigQueryGraphStore(
            project_id="proj",
            dataset_id="ds",
            graph_name="test_graph",
            client=client,
        )
        assert store.schema.graph_name == "test_graph"

    def test_query(self):
        client = _make_mock_client()
        store = BigQueryGraphStore(
            project_id="proj",
            dataset_id="ds",
            graph_name="test_graph",
            client=client,
        )

        row_dict = {"col1": "val1"}
        job = MagicMock()
        job.result.return_value = [row_dict]
        client.query.return_value = job

        results = store.query("GRAPH test_graph MATCH (n) RETURN n.id")
        assert len(results) == 1
        assert results[0]["col1"] == "val1"

    def test_get_schema_empty(self):
        client = _make_mock_client()
        store = BigQueryGraphStore(
            project_id="proj",
            dataset_id="ds",
            graph_name="test_graph",
            client=client,
        )
        schema_str = store.get_schema
        assert "test_graph" in schema_str

    def test_cleanup(self):
        client = _make_mock_client()
        store = BigQueryGraphStore(
            project_id="proj",
            dataset_id="ds",
            graph_name="test_graph",
            client=client,
        )
        store.cleanup()
        assert len(store.schema.nodes) == 0
        assert len(store.schema.edges) == 0
