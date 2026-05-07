from __future__ import annotations

import json
from enum import Enum
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.language_models import BaseLanguageModel
from langchain_core.load import dumps
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import Field

from langchain_bigquery.graph_store import BigQueryGraphStore

from .graph_utils import extract_gql
from .prompts import DEFAULT_GQL_TEMPLATE, DEFAULT_GQL_TEMPLATE_PART1


class DistanceStrategy(Enum):
    COSINE = 1
    EUCLIDEAN = 2


_DISTANCE_FUNCTIONS = {
    DistanceStrategy.COSINE: "COSINE_DISTANCE",
    DistanceStrategy.EUCLIDEAN: "EUCLIDEAN_DISTANCE",
}

GQL_GENERATION_PROMPT = PromptTemplate(
    template=DEFAULT_GQL_TEMPLATE,
    input_variables=["question", "schema"],
)


def _convert_to_doc(data: dict[str, Any]) -> Document:
    content = dumps(data)
    return Document(page_content=content, metadata={})


def _get_graph_name_from_schema(schema: str) -> str:
    name = json.loads(schema)["Name of graph"]
    return ".".join("`" + part + "`" for part in name.split("."))


# ---------------------------------------------------------------------------
# BigQueryGraphTextToGQLRetriever
# ---------------------------------------------------------------------------


class BigQueryGraphTextToGQLRetriever(BaseRetriever):
    """Translates natural language queries to GQL and executes them
    against a BigQueryGraphStore.

    If examples are provided, uses semantic similarity to select
    few-shot examples for GQL generation.
    """

    graph_store: BigQueryGraphStore = Field(exclude=True)
    k: int = 10
    llm: Optional[BaseLanguageModel] = None
    selector: Optional[SemanticSimilarityExampleSelector] = None

    @classmethod
    def from_params(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        embedding_service: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> BigQueryGraphTextToGQLRetriever:
        if llm is None:
            raise ValueError("`llm` cannot be None")
        selector = None
        if embedding_service is not None:
            selector = SemanticSimilarityExampleSelector.from_examples(
                [], embedding_service, InMemoryVectorStore, k=2
            )
        return cls(llm=llm, selector=selector, **kwargs)

    def __duplicate_braces_in_string(self, text: str) -> str:
        text = text.replace("{", "{{")
        text = text.replace("}", "}}")
        return text

    def add_example(self, question: str, gql: str) -> None:
        if self.selector is None:
            raise ValueError("`selector` cannot be None")
        self.selector.add_example(
            {
                "input": question,
                "query": self.__duplicate_braces_in_string(gql),
            }
        )

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.llm is None:
            raise ValueError("`llm` cannot be None")

        gql_chain: RunnableSequence
        if self.selector is None:
            gql_chain = RunnableSequence(
                GQL_GENERATION_PROMPT | self.llm | StrOutputParser()
            )
        else:
            few_shot_prompt = FewShotPromptTemplate(
                example_selector=self.selector,
                example_prompt=PromptTemplate.from_template(
                    "Question: {input}\nGQL Query: {query}"
                ),
                prefix="Create an ISO GQL query for the question using the schema.",
                suffix=DEFAULT_GQL_TEMPLATE_PART1,
                input_variables=["question", "schema"],
            )
            gql_chain = RunnableSequence(
                few_shot_prompt | self.llm | StrOutputParser()
            )

        gql_query = extract_gql(
            gql_chain.invoke(
                {
                    "question": question,
                    "schema": self.graph_store.get_schema,
                }
            )
        )

        responses = self.graph_store.query(gql_query)[: self.k]

        return [_convert_to_doc(response) for response in responses]


# ---------------------------------------------------------------------------
# BigQueryGraphVectorContextRetriever
# ---------------------------------------------------------------------------


class BigQueryGraphVectorContextRetriever(BaseRetriever):
    """Retriever that performs vector search on nodes in a BigQueryGraphStore.

    If expand_by_hops is provided, the nodes (and edges) at a distance up to
    expand_by_hops will also be returned.
    """

    graph_store: BigQueryGraphStore = Field(exclude=True)
    embedding_service: Optional[Embeddings] = None
    label_expr: str = "%"
    """A label expression for the nodes to search."""
    return_properties_list: List[str] = []
    """The list of properties to return."""
    embeddings_column: str = "embedding"
    """The name of the column that stores embeddings."""
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    """Distance function to use for vector similarity."""
    top_k: int = 3
    """Number of vector similarity matches to return."""
    expand_by_hops: int = -1
    """Number of hops to traverse to expand graph results."""
    k: int = 10
    """Number of graph results to return."""

    @classmethod
    def from_params(
        cls, embedding_service: Embeddings, **kwargs: Any
    ) -> BigQueryGraphVectorContextRetriever:
        if embedding_service is None:
            raise ValueError("`embedding_service` cannot be None")
        return cls(embedding_service=embedding_service, **kwargs)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.embedding_service is None:
            raise ValueError("`embedding_service` cannot be None")

        options = 0
        if self.return_properties_list:
            options += 1
        if self.expand_by_hops != -1:
            options += 1
        if options != 1:
            raise ValueError(
                "One and only one of `return_properties_list` or `expand_by_hops` must be provided."
            )

    def __clean_element(self, element: dict[str, Any]) -> None:
        keys_to_remove = [
            "source_node_identifier",
            "destination_node_identifier",
            "identifier",
        ]
        for key in keys_to_remove:
            if key in element:
                del element[key]

        if "properties" in element and self.embeddings_column in element["properties"]:
            del element["properties"][self.embeddings_column]

    def _get_node_table_fqn(self, label: str) -> str:
        if label == "%":
            for node in self.graph_store.schema.nodes.values():
                if self.embeddings_column in node.types:
                    return "`{}`.`{}`".format(
                        self.graph_store._dataset_id, node.base_table_name
                    )
            raise ValueError(
                "No node table with column `%s` found" % self.embeddings_column
            )
        node_schema = self.graph_store.schema.get_node_schema(label)
        if node_schema is None:
            raise ValueError("No node schema found for label: `%s`" % label)
        return "`{}`.`{}`".format(
            self.graph_store._dataset_id, node_schema.base_table_name
        )

    def _get_relevant_documents(
        self, question: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        schema = self.graph_store.get_schema
        graph_name = _get_graph_name_from_schema(schema)

        if self.embedding_service is None:
            raise ValueError("`embedding_service` cannot be None")
        query_embeddings = self.embedding_service.embed_query(question)

        distance_fn = _DISTANCE_FUNCTIONS.get(
            self.distance_strategy, "COSINE_DISTANCE"
        )
        embeddings_array = "ARRAY[{}]".format(
            ",".join(map(str, query_embeddings))
        )
        table_fqn = self._get_node_table_fqn(self.label_expr)

        if self.return_properties_list:
            return_cols = ", ".join(
                "`{}`".format(p) for p in self.return_properties_list
            )
            sql = (
                "SELECT {return_cols} FROM {table_fqn}\n"
                "WHERE `{emb_col}` IS NOT NULL\n"
                "ORDER BY {dist_fn}(`{emb_col}`, {emb_array})\n"
                "LIMIT {top_k}"
            ).format(
                return_cols=return_cols,
                table_fqn=table_fqn,
                emb_col=self.embeddings_column,
                dist_fn=distance_fn,
                emb_array=embeddings_array,
                top_k=self.top_k,
            )
            responses = self.graph_store.query(sql)[: self.k]
            return [_convert_to_doc(r) for r in responses]

        vector_sql = (
            "SELECT `id` FROM {table_fqn}\n"
            "WHERE `{emb_col}` IS NOT NULL\n"
            "ORDER BY {dist_fn}(`{emb_col}`, {emb_array})\n"
            "LIMIT {top_k}"
        ).format(
            table_fqn=table_fqn,
            emb_col=self.embeddings_column,
            dist_fn=distance_fn,
            emb_array=embeddings_array,
            top_k=self.top_k,
        )
        id_rows = self.graph_store.query(vector_sql)
        if not id_rows:
            return []

        id_list = ", ".join("'{}'".format(r["id"]) for r in id_rows)

        if self.expand_by_hops == 0:
            gql_query = (
                "GRAPH {graph_name}\n"
                "MATCH (node:{label_expr})\n"
                "WHERE node.id IN ({id_list})\n"
                "RETURN TO_JSON(node) as path"
            ).format(
                graph_name=graph_name,
                label_expr=self.label_expr,
                id_list=id_list,
            )
        elif self.expand_by_hops > 0:
            gql_query = (
                "GRAPH {graph_name}\n"
                "MATCH (node:{label_expr})\n"
                "WHERE node.id IN ({id_list})\n"
                "RETURN node\n"
                "NEXT\n"
                "MATCH p = TRAIL (node) -[]-{{0,{hops}}} ()\n"
                "RETURN TO_JSON(p) as path"
            ).format(
                graph_name=graph_name,
                label_expr=self.label_expr,
                id_list=id_list,
                hops=self.expand_by_hops,
            )
        else:
            raise ValueError(
                "Either `return_properties_list` or `expand_by_hops` must be provided."
            )

        responses = self.graph_store.query(gql_query)[: self.k]

        documents = []
        if self.expand_by_hops > 0:
            for response in responses:
                path_data = response.get("path")
                if path_data is None:
                    continue
                if isinstance(path_data, str):
                    elements = json.loads(path_data)
                elif hasattr(path_data, "serialize"):
                    elements = json.loads(path_data.serialize())
                else:
                    elements = path_data

                if isinstance(elements, list):
                    for element in elements:
                        if isinstance(element, dict):
                            self.__clean_element(element)

                response["path"] = elements
                content = dumps(response["path"])
                documents.append(Document(page_content=content, metadata={}))
        else:
            for response in responses:
                documents.append(_convert_to_doc(response))

        return documents
