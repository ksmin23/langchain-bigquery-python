from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from google.cloud import bigquery
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.graph_store import GraphStore
from requests.structures import CaseInsensitiveDict

MUTATION_BATCH_SIZE = 500
NODE_KIND = "NODE"
EDGE_KIND = "EDGE"

# ---------------------------------------------------------------------------
# Graph document utilities (adapted from Spanner implementation)
# ---------------------------------------------------------------------------


class NodeWrapper:
    def __init__(self, node: Node):
        self.node = node

    def __hash__(self):
        return hash(self.node.id)

    def __eq__(self, other: Any):
        if isinstance(other, NodeWrapper):
            return self.node.id == other.node.id
        return False


class EdgeWrapper:
    def __init__(self, edge: Relationship):
        self.edge = edge

    def __hash__(self):
        return hash((self.edge.source.id, self.edge.target.id, self.edge.type))

    def __eq__(self, other: Any):
        if isinstance(other, EdgeWrapper):
            return (
                self.edge.source.id == other.edge.source.id
                and self.edge.target.id == other.edge.target.id
                and self.edge.type == other.edge.type
            )
        return False


def partition_graph_docs(
    graph_documents: List[GraphDocument],
) -> Tuple[dict, dict]:
    nodes: CaseInsensitiveDict[dict[NodeWrapper, Node]] = CaseInsensitiveDict()
    edges: CaseInsensitiveDict[dict[EdgeWrapper, Relationship]] = CaseInsensitiveDict()
    for doc in graph_documents:
        for node in doc.nodes:
            ns = nodes.setdefault(node.type, dict())
            nw = NodeWrapper(node)
            if nw in ns:
                ns[nw].properties.update(node.properties)
            else:
                ns[nw] = node
        for edge in doc.relationships:
            edge_name = "{}_{}_{}".format(edge.source.type, edge.type, edge.target.type)
            es = edges.setdefault(edge_name, dict())
            ew = EdgeWrapper(edge)
            if ew in es:
                es[ew].properties.update(edge.properties)
            else:
                es[ew] = edge
    return (
        {name: [n for _, n in ns.items()] for name, ns in nodes.items()},
        {name: [e for _, e in es.items()] for name, es in edges.items()},
    )


class _GraphDocUtil:
    @staticmethod
    def is_valid_identifier(s: str) -> bool:
        return re.match(r"^[a-z][a-z0-9_]{0,127}$", s, re.IGNORECASE) is not None

    @staticmethod
    def to_identifier(s: str) -> str:
        return "`" + s + "`"

    @staticmethod
    def to_identifiers(s: List[str]) -> Iterable[str]:
        return map(_GraphDocUtil.to_identifier, s)

    @staticmethod
    def fixup_identifier(s: str) -> str:
        return re.sub("[{}]".format(string.whitespace + string.punctuation), "_", s)

    @staticmethod
    def fixup_graph_documents(graph_documents: List[GraphDocument]) -> None:
        for doc in graph_documents:
            for node in doc.nodes:
                _GraphDocUtil.fixup_element(node)
            for edge in doc.relationships:
                _GraphDocUtil.fixup_element(edge)

    @staticmethod
    def fixup_element(element: Union[Node, Relationship]) -> None:
        element.type = _GraphDocUtil.fixup_identifier(element.type)
        should_ignore = lambda v: v is None or (isinstance(v, list) and len(v) == 0)
        element.properties = {
            k: v for k, v in element.properties.items() if not should_ignore(v)
        }
        if isinstance(element, Relationship):
            element.source.type = _GraphDocUtil.fixup_identifier(element.source.type)
            element.target.type = _GraphDocUtil.fixup_identifier(element.target.type)


# ---------------------------------------------------------------------------
# Type mapping helpers
# ---------------------------------------------------------------------------

_PYTHON_TO_BQ_TYPE: Dict[type, str] = {
    str: "STRING",
    int: "INT64",
    float: "FLOAT64",
    bool: "BOOL",
    bytes: "BYTES",
    list: "ARRAY<FLOAT64>",
}


def _value_to_bq_type(value: Any) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], float):
            return "ARRAY<FLOAT64>"
        return "ARRAY<STRING>"
    return _PYTHON_TO_BQ_TYPE.get(type(value), "STRING")


def _bq_schema_type_to_str(col_type: str) -> str:
    return col_type


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


class Label:
    def __init__(self, name: str, prop_names: set[str]):
        self.name = name
        self.prop_names = prop_names


# ---------------------------------------------------------------------------
# NodeReference
# ---------------------------------------------------------------------------


class NodeReference:
    def __init__(self, node_name: str, node_keys: List[str], edge_keys: List[str]):
        self.node_name = node_name
        self.node_keys = node_keys
        self.edge_keys = edge_keys


# ---------------------------------------------------------------------------
# ElementSchema
# ---------------------------------------------------------------------------


class ElementSchema:
    NODE_KEY_COLUMN_NAME: str = "id"
    TARGET_NODE_KEY_COLUMN_NAME: str = "target_id"
    DYNAMIC_PROPERTY_COLUMN_NAME: str = "properties"
    DYNAMIC_LABEL_COLUMN_NAME: str = "label"

    name: str
    kind: str
    key_columns: List[str]
    base_table_name: str
    labels: List[str]
    properties: CaseInsensitiveDict[str]
    types: CaseInsensitiveDict[str]
    source: NodeReference
    target: NodeReference

    def is_dynamic_schema(self) -> bool:
        return self.types.get(ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME) == "JSON"

    # -- factory methods for nodes --

    @staticmethod
    def make_node_schema(
        node_type: str,
        node_label: str,
        graph_name: str,
        dataset_id: str,
        property_types: CaseInsensitiveDict,
    ) -> ElementSchema:
        node = ElementSchema()
        node.types = property_types
        node.properties = CaseInsensitiveDict({prop: prop for prop in node.types})
        node.labels = [node_label]
        node.base_table_name = "%s_%s" % (graph_name, node_label)
        node.name = node_type
        node.kind = NODE_KIND
        node.key_columns = [ElementSchema.NODE_KEY_COLUMN_NAME]
        return node

    @staticmethod
    def from_static_nodes(
        name: str, nodes: List[Node], graph_schema: BigQueryGraphSchema
    ) -> ElementSchema:
        if not nodes:
            raise ValueError("The list of nodes should not be empty")
        types = CaseInsensitiveDict(
            {k: _value_to_bq_type(v) for n in nodes for k, v in n.properties.items()}
        )
        if ElementSchema.NODE_KEY_COLUMN_NAME in types:
            raise ValueError(
                "Node properties should not contain property named: `%s`"
                % ElementSchema.NODE_KEY_COLUMN_NAME
            )
        types[ElementSchema.NODE_KEY_COLUMN_NAME] = _value_to_bq_type(nodes[0].id)
        return ElementSchema.make_node_schema(
            name, name, graph_schema.graph_name, graph_schema.dataset_id, types
        )

    @staticmethod
    def from_dynamic_nodes(
        name: str, nodes: List[Node], graph_schema: BigQueryGraphSchema
    ) -> ElementSchema:
        if not nodes:
            raise ValueError("The list of nodes should not be empty")
        types = CaseInsensitiveDict(
            {
                ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME: "JSON",
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME: "STRING",
                ElementSchema.NODE_KEY_COLUMN_NAME: _value_to_bq_type(nodes[0].id),
            }
        )
        types.update(
            CaseInsensitiveDict(
                {
                    k: _value_to_bq_type(v)
                    for n in nodes
                    for k, v in n.properties.items()
                    if k in graph_schema.static_node_properties
                }
            )
        )
        return ElementSchema.make_node_schema(
            NODE_KIND, NODE_KIND, graph_schema.graph_name, graph_schema.dataset_id, types
        )

    # -- factory methods for edges --

    @staticmethod
    def make_edge_schema(
        edge_type: str,
        edge_label: str,
        graph_schema: BigQueryGraphSchema,
        key_columns: List[str],
        property_types: CaseInsensitiveDict,
        source_node_type: str,
        target_node_type: str,
    ) -> ElementSchema:
        edge = ElementSchema()
        edge.types = property_types
        edge.properties = CaseInsensitiveDict({prop: prop for prop in edge.types})
        edge.labels = [edge_label]
        edge.base_table_name = "%s_%s" % (graph_schema.graph_name, edge_type)
        edge.key_columns = key_columns
        edge.name = edge_type
        edge.kind = EDGE_KIND

        source_node_schema = graph_schema.get_node_schema(
            graph_schema.node_type_name(source_node_type)
        )
        if source_node_schema is None:
            raise ValueError("No source node schema `%s` found" % source_node_type)

        target_node_schema = graph_schema.get_node_schema(
            graph_schema.node_type_name(target_node_type)
        )
        if target_node_schema is None:
            raise ValueError("No target node schema `%s` found" % target_node_type)

        edge.source = NodeReference(
            source_node_schema.name,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.NODE_KEY_COLUMN_NAME],
        )
        edge.target = NodeReference(
            target_node_schema.name,
            [ElementSchema.NODE_KEY_COLUMN_NAME],
            [ElementSchema.TARGET_NODE_KEY_COLUMN_NAME],
        )
        return edge

    @staticmethod
    def from_static_edges(
        name: str,
        edges: List[Relationship],
        graph_schema: BigQueryGraphSchema,
    ) -> ElementSchema:
        if not edges:
            raise ValueError("The list of edges should not be empty")
        types = CaseInsensitiveDict(
            {k: _value_to_bq_type(v) for e in edges for k, v in e.properties.items()}
        )
        for col_name in [
            ElementSchema.NODE_KEY_COLUMN_NAME,
            ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
        ]:
            if col_name in types:
                raise ValueError(
                    "Edge properties should not contain property named: `%s`" % col_name
                )
        types[ElementSchema.NODE_KEY_COLUMN_NAME] = _value_to_bq_type(
            edges[0].source.id
        )
        types[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = _value_to_bq_type(
            edges[0].target.id
        )
        return ElementSchema.make_edge_schema(
            name,
            edges[0].type,
            graph_schema,
            [
                ElementSchema.NODE_KEY_COLUMN_NAME,
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
            ],
            types,
            edges[0].source.type,
            edges[0].target.type,
        )

    @staticmethod
    def from_dynamic_edges(
        name: str,
        edges: List[Relationship],
        graph_schema: BigQueryGraphSchema,
    ) -> ElementSchema:
        if not edges:
            raise ValueError("The list of edges should not be empty")
        types = CaseInsensitiveDict(
            {
                ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME: "JSON",
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME: "STRING",
                ElementSchema.NODE_KEY_COLUMN_NAME: _value_to_bq_type(
                    edges[0].source.id
                ),
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME: _value_to_bq_type(
                    edges[0].target.id
                ),
            }
        )
        types.update(
            CaseInsensitiveDict(
                {
                    k: _value_to_bq_type(v)
                    for e in edges
                    for k, v in e.properties.items()
                    if k in graph_schema.static_edge_properties
                }
            )
        )
        return ElementSchema.make_edge_schema(
            EDGE_KIND,
            EDGE_KIND,
            graph_schema,
            [
                ElementSchema.NODE_KEY_COLUMN_NAME,
                ElementSchema.TARGET_NODE_KEY_COLUMN_NAME,
                ElementSchema.DYNAMIC_LABEL_COLUMN_NAME,
            ],
            types,
            edges[0].source.type,
            edges[0].target.type,
        )

    @staticmethod
    def from_info_schema(
        element_schema: Dict[str, Any],
        decl_by_types: CaseInsensitiveDict,
        kind: str,
    ) -> ElementSchema:
        element = ElementSchema()
        element.name = element_schema["name"]
        element.kind = kind

        element.key_columns = element_schema["key_columns"]
        element.base_table_name = element_schema["base_table_name"]
        element.labels = element_schema["label_names"]

        element.properties = CaseInsensitiveDict(
            {
                prop_def["property_declaration_name"]: prop_def["value_expression_sql"]
                for prop_def in element_schema.get("property_definitions", [])
                if prop_def["property_declaration_name"] in decl_by_types
            }
        )
        element.types = CaseInsensitiveDict(
            {decl: decl_by_types[decl] for decl in element.properties.keys()}
        )

        if element.kind == EDGE_KIND:
            element.source = NodeReference(
                element_schema["source_node_table"]["node_table_name"],
                element_schema["source_node_table"]["node_table_columns"],
                element_schema["source_node_table"]["edge_table_columns"],
            )
            element.target = NodeReference(
                element_schema["destination_node_table"]["node_table_name"],
                element_schema["destination_node_table"]["node_table_columns"],
                element_schema["destination_node_table"]["edge_table_columns"],
            )
        return element

    # -- DDL generation --

    def to_ddl(self, graph_schema: BigQueryGraphSchema) -> List[str]:
        to_id = _GraphDocUtil.to_identifier
        fqn = lambda t: "`{}`.{}".format(graph_schema.dataset_id, to_id(t))

        col_defs = ",\n    ".join(
            "{} {}".format(to_id(n), t) for n, t in self.types.items()
        )
        pk = ",".join(_GraphDocUtil.to_identifiers(self.key_columns))
        table_fqn = fqn(self.base_table_name)

        ddls = [
            "CREATE TABLE IF NOT EXISTS {} (\n    {},\n    PRIMARY KEY ({}) NOT ENFORCED\n)".format(
                table_fqn, col_defs, pk
            )
        ]
        for n, t in self.types.items():
            if n not in self.key_columns:
                ddls.append(
                    "ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}".format(
                        table_fqn, to_id(n), t
                    )
                )
        return ddls

    def to_graph_element_ddl(self, graph_schema: BigQueryGraphSchema) -> str:
        to_id = _GraphDocUtil.to_identifier
        fqn = lambda t: "`{}`.{}".format(graph_schema.dataset_id, to_id(t))

        def get_ref_table(name: str) -> str:
            ns = graph_schema.nodes.get(name)
            if ns is None:
                raise ValueError("No node schema `%s` found" % name)
            return ns.base_table_name

        lines = [
            "{} AS {}".format(fqn(self.base_table_name), to_id(self.name)),
            "KEY({})".format(",".join(_GraphDocUtil.to_identifiers(self.key_columns))),
        ]
        if self.kind == EDGE_KIND:
            lines.append(
                "SOURCE KEY({}) REFERENCES {}({})".format(
                    ",".join(_GraphDocUtil.to_identifiers(self.source.edge_keys)),
                    to_id(self.source.node_name),
                    ",".join(_GraphDocUtil.to_identifiers(self.source.node_keys)),
                )
            )
            lines.append(
                "DESTINATION KEY({}) REFERENCES {}({})".format(
                    ",".join(_GraphDocUtil.to_identifiers(self.target.edge_keys)),
                    to_id(self.target.node_name),
                    ",".join(_GraphDocUtil.to_identifiers(self.target.node_keys)),
                )
            )
        for label in self.labels:
            label_obj = graph_schema.labels.get(label)
            if label_obj is None:
                continue
            prop_defs = []
            for k, v in self.properties.items():
                if k in label_obj.prop_names:
                    col_type = self.types.get(k, "")
                    if col_type.startswith("ARRAY"):
                        continue
                    if k != v:
                        prop_defs.append("{} AS {}".format(v, to_id(k)))
                    else:
                        prop_defs.append(to_id(k))
            lines.append(
                "LABEL {} PROPERTIES({})".format(to_id(label), ", ".join(prop_defs))
            )
        return "\n    ".join(lines)

    def evolve(self, new_schema: ElementSchema) -> List[str]:
        if self.kind != new_schema.kind:
            raise ValueError(
                "Schema `{}` kind mismatch: got {}, expected {}".format(
                    self.name, new_schema.kind, self.kind
                )
            )
        if self.key_columns != new_schema.key_columns:
            raise ValueError(
                "Schema `{}` key mismatch: got {}, expected {}".format(
                    self.name, new_schema.key_columns, self.key_columns
                )
            )
        if self.base_table_name.casefold() != new_schema.base_table_name.casefold():
            raise ValueError(
                "Schema `{}` table name mismatch: got {}, expected {}".format(
                    self.name, new_schema.base_table_name, self.base_table_name
                )
            )

        if self.name == new_schema.name:
            for k, v in new_schema.properties.items():
                if k in self.properties:
                    old_v = self.properties[k].strip("`").casefold()
                    new_v = v.strip("`").casefold()
                    if old_v != new_v:
                        raise ValueError(
                            "Property `{}` definition mismatch: got {}, expected {}".format(
                                k, v, self.properties[k]
                            )
                        )
        for k, v in new_schema.types.items():
            if k in self.types and self.types[k] != v:
                raise ValueError(
                    "Property `{}` type mismatch: got {}, expected {}".format(
                        k, v, self.types[k]
                    )
                )

        to_id = _GraphDocUtil.to_identifier
        fqn = "`{}`.{}".format("_dataset_placeholder_", to_id(self.base_table_name))
        ddls = [
            "ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}".format(fqn, to_id(n), t)
            for n, t in new_schema.types.items()
            if n not in self.properties
        ]
        self.properties.update(new_schema.properties)
        self.types.update(new_schema.types)
        return ddls

    # -- data insertion helpers --

    def build_merge_nodes(
        self, nodes: List[Node], dataset_id: str
    ) -> Generator[Tuple[str, List], None, None]:
        if not nodes:
            return
        to_id = _GraphDocUtil.to_identifier
        fqn = "`{}`.{}".format(dataset_id, to_id(self.base_table_name))

        for node in nodes:
            props: Dict[str, Any] = node.properties.copy()
            props[ElementSchema.NODE_KEY_COLUMN_NAME] = node.id

            if self.is_dynamic_schema():
                dynamic = {
                    k: v for k, v in node.properties.items() if k not in self.types
                }
                if dynamic:
                    props[ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME] = json.dumps(
                        dynamic, ensure_ascii=False
                    )
                props[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = node.type

            columns = sorted(k for k in props if k in self.types)
            select_parts = []
            params = []
            for i, col in enumerate(columns):
                pname = f"p{i}"
                select_parts.append(f"@{pname} AS {to_id(col)}")
                params.append(
                    _make_bq_param(pname, self.types[col], props[col])
                )

            set_clause = ", ".join(
                f"T.{to_id(c)} = S.{to_id(c)}"
                for c in columns
                if c not in self.key_columns
            )
            all_cols = ", ".join(to_id(c) for c in columns)
            all_vals = ", ".join(f"S.{to_id(c)}" for c in columns)

            sql = (
                f"MERGE INTO {fqn} AS T\n"
                f"USING (SELECT {', '.join(select_parts)}) AS S\n"
                f"ON T.{to_id(self.key_columns[0])} = S.{to_id(self.key_columns[0])}\n"
            )
            if set_clause:
                sql += f"WHEN MATCHED THEN UPDATE SET {set_clause}\n"
            sql += f"WHEN NOT MATCHED THEN INSERT ({all_cols}) VALUES ({all_vals})"
            yield sql, params

    def build_merge_edges(
        self, edges: List[Relationship], dataset_id: str
    ) -> Generator[Tuple[str, List], None, None]:
        if not edges:
            return
        to_id = _GraphDocUtil.to_identifier
        fqn = "`{}`.{}".format(dataset_id, to_id(self.base_table_name))

        for edge in edges:
            props: Dict[str, Any] = edge.properties.copy()
            props[ElementSchema.NODE_KEY_COLUMN_NAME] = edge.source.id
            props[ElementSchema.TARGET_NODE_KEY_COLUMN_NAME] = edge.target.id

            if self.is_dynamic_schema():
                dynamic = {
                    k: v for k, v in edge.properties.items() if k not in self.types
                }
                if dynamic:
                    props[ElementSchema.DYNAMIC_PROPERTY_COLUMN_NAME] = json.dumps(
                        dynamic, ensure_ascii=False
                    )
                props[ElementSchema.DYNAMIC_LABEL_COLUMN_NAME] = edge.type

            columns = sorted(k for k in props if k in self.types)
            select_parts = []
            params = []
            for i, col in enumerate(columns):
                pname = f"p{i}"
                select_parts.append(f"@{pname} AS {to_id(col)}")
                params.append(
                    _make_bq_param(pname, self.types[col], props[col])
                )

            key_on = " AND ".join(
                f"T.{to_id(k)} = S.{to_id(k)}" for k in self.key_columns
            )
            non_key = [c for c in columns if c not in self.key_columns]
            set_clause = ", ".join(
                f"T.{to_id(c)} = S.{to_id(c)}" for c in non_key
            )
            all_cols = ", ".join(to_id(c) for c in columns)
            all_vals = ", ".join(f"S.{to_id(c)}" for c in columns)

            sql = (
                f"MERGE INTO {fqn} AS T\n"
                f"USING (SELECT {', '.join(select_parts)}) AS S\n"
                f"ON {key_on}\n"
            )
            if set_clause:
                sql += f"WHEN MATCHED THEN UPDATE SET {set_clause}\n"
            sql += f"WHEN NOT MATCHED THEN INSERT ({all_cols}) VALUES ({all_vals})"
            yield sql, params


def _make_bq_param(name: str, bq_type: str, value: Any):
    if bq_type.startswith("ARRAY<"):
        inner = bq_type[6:-1]
        return bigquery.ArrayQueryParameter(name, inner, value if value else [])
    return bigquery.ScalarQueryParameter(name, bq_type, value)


# ---------------------------------------------------------------------------
# INFORMATION_SCHEMA normalization (camelCase → snake_case)
# ---------------------------------------------------------------------------


def _normalize_info_schema(info_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BigQuery INFORMATION_SCHEMA camelCase metadata to the
    snake_case format expected by ``BigQueryGraphSchema.from_information_schema``
    and ``ElementSchema.from_info_schema``.

    If the metadata already uses snake_case keys it is returned unchanged.
    """
    if "node_tables" in info_schema:
        return info_schema

    all_properties: Dict[str, str] = {}

    def _normalize_element(element: Dict[str, Any]) -> Dict[str, Any]:
        label_and_props = element.get("labelAndProperties", [])
        label_names = [lp["label"] for lp in label_and_props]

        property_definitions = []
        for lp in label_and_props:
            for prop in lp.get("properties", []):
                property_definitions.append({
                    "property_declaration_name": prop["name"],
                    "value_expression_sql": prop.get("expression", ""),
                })
                if "dataType" in prop:
                    all_properties[prop["name"]] = prop["dataType"].get(
                        "typeKind", "STRING"
                    )

        result: Dict[str, Any] = {
            "name": element["name"],
            "key_columns": element.get("keyColumns", []),
            "base_table_name": element.get("dataSourceTable", {}).get(
                "tableId", ""
            ),
            "label_names": label_names,
            "property_definitions": property_definitions,
        }

        if "sourceNodeReference" in element:
            src = element["sourceNodeReference"]
            result["source_node_table"] = {
                "node_table_name": src["nodeTable"],
                "node_table_columns": src["nodeTableColumns"],
                "edge_table_columns": src["edgeTableColumns"],
            }
        if "destinationNodeReference" in element:
            dst = element["destinationNodeReference"]
            result["destination_node_table"] = {
                "node_table_name": dst["nodeTable"],
                "node_table_columns": dst["nodeTableColumns"],
                "edge_table_columns": dst["edgeTableColumns"],
            }

        return result

    return {
        "node_tables": [
            _normalize_element(n)
            for n in info_schema.get("nodeTables", [])
        ],
        "edge_tables": [
            _normalize_element(e)
            for e in info_schema.get("edgeTables", [])
        ],
        "property_declarations": [
            {"name": name, "type": type_kind}
            for name, type_kind in all_properties.items()
        ],
    }


# ---------------------------------------------------------------------------
# BigQueryGraphSchema
# ---------------------------------------------------------------------------


class BigQueryGraphSchema:
    GRAPH_INFORMATION_SCHEMA_QUERY = """
    SELECT property_graph_metadata_json
    FROM `{dataset}`.INFORMATION_SCHEMA.PROPERTY_GRAPHS
    WHERE property_graph_name = '{graph_name}'
    """

    def __init__(
        self,
        graph_name: str,
        dataset_id: str,
        use_flexible_schema: bool = False,
        static_node_properties: List[str] = [],
        static_edge_properties: List[str] = [],
    ):
        if not _GraphDocUtil.is_valid_identifier(graph_name):
            raise ValueError("Graph name `{}` is not a valid identifier".format(graph_name))

        self.graph_name = graph_name
        self.dataset_id = dataset_id
        self.nodes: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.edges: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.node_tables: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.edge_tables: CaseInsensitiveDict[ElementSchema] = CaseInsensitiveDict({})
        self.labels: CaseInsensitiveDict[Label] = CaseInsensitiveDict({})
        self.properties: CaseInsensitiveDict[str] = CaseInsensitiveDict({})
        self.use_flexible_schema = use_flexible_schema
        self.static_node_properties = set(static_node_properties)
        self.static_edge_properties = set(static_edge_properties)

    def node_type_name(self, name: str) -> str:
        return NODE_KIND if self.use_flexible_schema else name

    def edge_type_name(self, name: str) -> str:
        return EDGE_KIND if self.use_flexible_schema else name

    def get_node_schema(self, name: str) -> Optional[ElementSchema]:
        return self.nodes.get(name, None)

    def get_edge_schema(self, name: str) -> Optional[ElementSchema]:
        return self.edges.get(name, None)

    def evolve(self, graph_documents: List[GraphDocument]) -> List[str]:
        nodes, edges = partition_graph_docs(graph_documents)
        ddls: List[str] = []
        for k, ns in nodes.items():
            node_schema = (
                ElementSchema.from_static_nodes(k, ns, self)
                if not self.use_flexible_schema
                else ElementSchema.from_dynamic_nodes(k, ns, self)
            )
            ddls.extend(self._update_node_schema(node_schema))
            self._update_labels_and_properties(node_schema)

        for k, es in edges.items():
            edge_schema = (
                ElementSchema.from_static_edges(k, es, self)
                if not self.use_flexible_schema
                else ElementSchema.from_dynamic_edges(k, es, self)
            )
            ddls.extend(self._update_edge_schema(edge_schema))
            self._update_labels_and_properties(edge_schema)

        if ddls:
            ddls.append(self.to_graph_ddl())
        return ddls

    def from_information_schema(self, info_schema: Dict[str, Any]) -> None:
        info_schema = _normalize_info_schema(info_schema)
        property_decls = info_schema.get("property_declarations", [])
        decl_by_types = CaseInsensitiveDict(
            {decl["name"]: decl["type"] for decl in property_decls}
        )
        for node in info_schema.get("node_tables", []):
            node_schema = ElementSchema.from_info_schema(node, decl_by_types, kind=NODE_KIND)
            self._update_node_schema(node_schema)
            self._update_labels_and_properties(node_schema)

        for edge in info_schema.get("edge_tables", []):
            edge_schema = ElementSchema.from_info_schema(edge, decl_by_types, kind=EDGE_KIND)
            self._update_edge_schema(edge_schema)
            self._update_labels_and_properties(edge_schema)

    def __repr__(self) -> str:
        node_labels = {label for node in self.nodes.values() for label in node.labels}
        edge_labels = {label for edge in self.edges.values() for label in edge.labels}

        triplets_per_label: CaseInsensitiveDict[List] = CaseInsensitiveDict({})
        for edge in self.edges.values():
            for label in edge.labels:
                source_node = self.get_node_schema(edge.source.node_name)
                target_node = self.get_node_schema(edge.target.node_name)
                if source_node and target_node:
                    triplets_per_label.setdefault(label, []).append(
                        (source_node, edge, target_node)
                    )

        return json.dumps(
            {
                "Name of graph": "{}.{}".format(self.dataset_id, self.graph_name),
                "Node properties per node label": {
                    label: [
                        {"name": name, "type": self.properties.get(name, "STRING")}
                        for name in sorted(self.labels[label].prop_names)
                    ]
                    for label in sorted(node_labels)
                },
                "Edge properties per edge label": {
                    label: [
                        {"name": name, "type": self.properties.get(name, "STRING")}
                        for name in sorted(self.labels[label].prop_names)
                    ]
                    for label in sorted(edge_labels)
                },
                "Possible edges per label": {
                    label: [
                        "(:{}) -[:{}]-> (:{})".format(
                            src_label, label, tgt_label
                        )
                        for (source, edge, target) in triplets
                        for src_label in source.labels
                        for tgt_label in target.labels
                    ]
                    for label, triplets in triplets_per_label.items()
                },
            },
            indent=2,
        )

    def to_graph_ddl(self) -> str:
        to_id = _GraphDocUtil.to_identifier

        ddl = "CREATE OR REPLACE PROPERTY GRAPH `{}`.{}".format(
            self.dataset_id, to_id(self.graph_name)
        )
        ddl += "\nNODE TABLES(\n  "
        ddl += ",\n  ".join(
            node.to_graph_element_ddl(self) for node in self.nodes.values()
        )
        ddl += "\n)"
        if self.edges:
            ddl += "\nEDGE TABLES(\n  "
            ddl += ",\n  ".join(
                edge.to_graph_element_ddl(self) for edge in self.edges.values()
            )
            ddl += "\n)"
        return ddl

    def _update_node_schema(self, node_schema: ElementSchema) -> List[str]:
        old_schema = self.nodes.get(node_schema.name)
        if old_schema is not None:
            ddls = old_schema.evolve(node_schema)
        elif node_schema.base_table_name in self.node_tables:
            ddls = self.node_tables[node_schema.base_table_name].evolve(node_schema)
        else:
            ddls = node_schema.to_ddl(self)
            self.node_tables[node_schema.base_table_name] = node_schema
        self.nodes[node_schema.name] = old_schema or node_schema
        return ddls

    def _update_edge_schema(self, edge_schema: ElementSchema) -> List[str]:
        old_schema = self.edges.get(edge_schema.name)
        if old_schema is not None:
            ddls = old_schema.evolve(edge_schema)
        elif edge_schema.base_table_name in self.edge_tables:
            ddls = self.edge_tables[edge_schema.base_table_name].evolve(edge_schema)
        else:
            ddls = edge_schema.to_ddl(self)
            self.edge_tables[edge_schema.base_table_name] = edge_schema
        self.edges[edge_schema.name] = old_schema or edge_schema
        return ddls

    def _update_labels_and_properties(self, element_schema: ElementSchema) -> None:
        for l in element_schema.labels:
            if l in self.labels:
                self.labels[l].prop_names.update(element_schema.properties.keys())
            else:
                self.labels[l] = Label(l, set(element_schema.properties.keys()))
        self.properties.update(element_schema.types)


# ---------------------------------------------------------------------------
# BigQueryGraphStore
# ---------------------------------------------------------------------------


class BigQueryGraphStore(GraphStore):
    """BigQuery Graph implementation of GraphStore."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        graph_name: str,
        client: Optional[bigquery.Client] = None,
        location: Optional[str] = None,
        use_flexible_schema: bool = False,
        static_node_properties: List[str] = [],
        static_edge_properties: List[str] = [],
    ):
        self._client = client or bigquery.Client(
            project=project_id, location=location
        )
        self._dataset_id = dataset_id
        self.schema = BigQueryGraphSchema(
            graph_name,
            dataset_id,
            use_flexible_schema,
            static_node_properties,
            static_edge_properties,
        )
        self.refresh_schema()

    def _execute_ddl(self, ddl: str) -> None:
        job = self._client.query(ddl)
        job.result()

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        if include_source:
            raise NotImplementedError("include_source is not supported yet")
        if baseEntityLabel:
            raise NotImplementedError("baseEntityLabel is not supported yet")

        _GraphDocUtil.fixup_graph_documents(graph_documents)

        ddls = self.schema.evolve(graph_documents)
        if ddls:
            for ddl in ddls:
                ddl = ddl.replace("_dataset_placeholder_", self._dataset_id)
                self._execute_ddl(ddl)
            self.refresh_schema()

        nodes, edges = partition_graph_docs(graph_documents)
        for name, elements in nodes.items():
            if not elements:
                continue
            node_schema = self.schema.get_node_schema(
                self.schema.node_type_name(name)
            )
            if node_schema is None:
                raise ValueError("Unknown node schema: `%s`" % name)
            for sql, params in node_schema.build_merge_nodes(
                elements, self._dataset_id
            ):
                job_config = bigquery.QueryJobConfig(query_parameters=params)
                job = self._client.query(sql, job_config=job_config)
                job.result()

        for name, elements in edges.items():
            if not elements:
                continue
            edge_schema = self.schema.get_edge_schema(
                self.schema.edge_type_name(name)
            )
            if edge_schema is None:
                raise ValueError("Unknown edge schema: `%s`" % name)
            for sql, params in edge_schema.build_merge_edges(
                elements, self._dataset_id
            ):
                job_config = bigquery.QueryJobConfig(query_parameters=params)
                job = self._client.query(sql, job_config=job_config)
                job.result()

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        bq_params = []
        for k, v in params.items():
            bq_params.append(
                _make_bq_param(k, _value_to_bq_type(v), v)
            )
        job_config = None
        if bq_params:
            job_config = bigquery.QueryJobConfig(query_parameters=bq_params)
        job = self._client.query(query, job_config=job_config)
        return [dict(row) for row in job.result()]

    @property
    def get_schema(self) -> str:
        return str(self.schema)

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        return json.loads(repr(self.schema))

    def get_ddl(self) -> str:
        return self.schema.to_graph_ddl()

    def refresh_schema(self) -> None:
        query = BigQueryGraphSchema.GRAPH_INFORMATION_SCHEMA_QUERY.format(
            dataset=self._dataset_id, graph_name=self.schema.graph_name
        )
        try:
            job = self._client.query(query)
            results = [dict(row) for row in job.result()]
        except Exception:
            return

        if not results:
            return
        if len(results) != 1:
            raise Exception(
                "Unexpected number of rows from information schema: {}".format(
                    len(results)
                )
            )
        self.schema.from_information_schema(results[0]["property_graph_metadata_json"])

    def cleanup(self) -> None:
        to_id = _GraphDocUtil.to_identifier
        fqn = lambda t: "`{}`.{}".format(self._dataset_id, to_id(t))

        try:
            self._execute_ddl(
                "DROP PROPERTY GRAPH IF EXISTS `{}`.{}".format(
                    self._dataset_id, to_id(self.schema.graph_name)
                )
            )
        except Exception:
            pass

        for edge in self.schema.edges.values():
            try:
                self._execute_ddl(
                    "DROP TABLE IF EXISTS {}".format(fqn(edge.base_table_name))
                )
            except Exception:
                pass
        for node in self.schema.nodes.values():
            try:
                self._execute_ddl(
                    "DROP TABLE IF EXISTS {}".format(fqn(node.base_table_name))
                )
            except Exception:
                pass

        self.schema = BigQueryGraphSchema(
            self.schema.graph_name,
            self._dataset_id,
            self.schema.use_flexible_schema,
        )
