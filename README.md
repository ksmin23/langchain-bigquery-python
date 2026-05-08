# BigQuery for LangChain

[![pypi](https://img.shields.io/pypi/v/langchain-bigquery.svg)](https://pypi.org/project/langchain-bigquery/)
[![versions](https://img.shields.io/pypi/pyversions/langchain-bigquery.svg)](https://pypi.org/project/langchain-bigquery/)

- [Client Library Documentation](docs/)
- [Product Documentation](https://cloud.google.com/bigquery)

## Quick Start

In order to use this library, you first need to go through the following steps:

1. [Select or create a Cloud Platform project.](https://console.cloud.google.com/project)
2. [Enable billing for your project.](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project)
3. [Enable the Google Cloud BigQuery API.](https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com)
4. [Setup Authentication.](https://googleapis.dev/python/google-api-core/latest/auth.html)

### Installation

Install this library in a [virtualenv](https://virtualenv.pypa.io/en/latest/) using pip. virtualenv is a tool to create isolated Python environments. The basic problem it addresses is one of dependencies and versions, and indirectly permissions.

With virtualenv, it's possible to install this library without needing system install permissions, and without clashing with the installed system dependencies.

#### Supported Python Versions

Python >= 3.10

#### Mac/Linux

```console
pip install virtualenv
virtualenv <your-env>
source <your-env>/bin/activate
<your-env>/bin/pip install langchain-bigquery
```

#### Windows

```console
pip install virtualenv
virtualenv <your-env>
<your-env>\Scripts\activate
<your-env>\Scripts\pip.exe install langchain-bigquery
```

## Prerequisites

### 1. Google Cloud Project Setup

- A Google Cloud project with [billing enabled](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project)
- The following APIs enabled:
  - [BigQuery API](https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com)
  - [Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com) (for embedding models)

```bash
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 2. Authentication

```bash
gcloud auth application-default login
```

To use Vertex AI with Application Default Credentials (no API key required):

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
```

### 3. IAM Permissions

The authenticated account needs the following roles (or equivalent permissions):

| Role | Purpose |
|------|---------|
| `roles/bigquery.dataEditor` | Create/delete tables, insert data |
| `roles/bigquery.jobUser` | Run queries (`VECTOR_SEARCH`, `SEARCH`) |
| `roles/bigquery.dataViewer` | Read table data and metadata |
| `roles/aiplatform.user` | Access Vertex AI embedding models |

```bash
PROJECT_ID=your-gcp-project-id
ACCOUNT=$(gcloud config get account)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:$ACCOUNT" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:$ACCOUNT" \
  --role="roles/bigquery.jobUser"
```

### 4. BigQuery Dataset

A BigQuery dataset must be created before using `BigQueryGraphStore` or `BigQueryHybridSearchVectorStore`. Tables and property graphs are created automatically, but the dataset is not.

```bash
bq mk --dataset --location=us-central1 YOUR_PROJECT_ID:YOUR_DATASET
```

Alternatively, grant `bigquery.datasets.create` permission to let the library create the dataset automatically.

## BigQuery Graph Store Usage

Use `BigQueryGraphStore` for storing and querying property graphs in BigQuery.

```python
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_bigquery import BigQueryGraphStore

store = BigQueryGraphStore(
    project_id="my-project",
    dataset_id="my_dataset",
    graph_name="knowledge_graph",
    location="us-central1",
)

# Define nodes and relationships
alice = Node(id="alice", type="Person", properties={"name": "Alice", "age": 30})
bob = Node(id="bob", type="Person", properties={"name": "Bob", "age": 25})
acme = Node(id="acme", type="Company", properties={"name": "Acme Corp"})

works_at = Relationship(source=alice, target=acme, type="WORKS_AT")
knows = Relationship(source=alice, target=bob, type="KNOWS")

doc = GraphDocument(
    nodes=[alice, bob, acme],
    relationships=[works_at, knows],
    source=Document(page_content="Alice works at Acme Corp and knows Bob."),
)

# This creates tables, the property graph, and inserts data
store.add_graph_documents([doc])

# Query with GQL
results = store.query(
    "GRAPH `my_dataset`.`knowledge_graph` MATCH (p:Person) RETURN p.name AS name"
)
print(results)
# [{'name': 'Alice'}, {'name': 'Bob'}]
```

## BigQuery Graph Vector Context Retriever Usage

Use `BigQueryGraphVectorContextRetriever` to perform vector similarity search on graph nodes with optional multi-hop neighborhood expansion.

```python
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_bigquery import BigQueryGraphVectorContextRetriever

embeddings = VertexAIEmbeddings(model="gemini-embedding-001")

# Return specific properties from matching nodes
retriever = BigQueryGraphVectorContextRetriever.from_params(
    graph_store=store,
    embedding_service=embeddings,
    label_expr="Person",
    embeddings_column="embedding",
    return_properties_list=["name", "age"],
    top_k=5,
)
docs = retriever.invoke("Who works at Acme?")
```

With multi-hop expansion:

```python
# Expand results by traversing 2 hops from matching nodes
retriever = BigQueryGraphVectorContextRetriever.from_params(
    graph_store=store,
    embedding_service=embeddings,
    label_expr="Person",
    embeddings_column="embedding",
    expand_by_hops=2,
    top_k=5,
)
docs = retriever.invoke("Tell me about Alice")
```

## BigQuery Graph Text-to-GQL Retriever Usage

Use `BigQueryGraphTextToGQLRetriever` to translate natural language questions into GQL queries using an LLM.

```python
from langchain_google_vertexai import ChatVertexAI
from langchain_bigquery import BigQueryGraphTextToGQLRetriever

llm = ChatVertexAI(model="gemini-2.5-flash")

retriever = BigQueryGraphTextToGQLRetriever.from_params(
    llm=llm,
    graph_store=store,
    k=10,
)

docs = retriever.invoke("Find all people who work at Acme Corp")
```

With few-shot examples for better GQL generation:

```python
from langchain_google_vertexai import VertexAIEmbeddings

retriever = BigQueryGraphTextToGQLRetriever.from_params(
    llm=llm,
    embedding_service=VertexAIEmbeddings(model="gemini-embedding-001"),
    graph_store=store,
)

retriever.add_example(
    question="Who works at Acme?",
    gql="GRAPH `my_dataset`.`knowledge_graph` MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: 'Acme Corp'}) RETURN p.name AS name",
)

docs = retriever.invoke("Which people are employed by Acme Corp?")
```

See the full [Graph RAG](samples/graph_rag.ipynb) tutorial.

## BigQuery Hybrid Search Usage

Use `BigQueryHybridSearchVectorStore` for hybrid (vector + full-text) search. Combines BigQuery `VECTOR_SEARCH()` (semantic similarity) with `SEARCH()` (full-text keyword matching) into a single retrieval step.

```python
from langchain_bigquery import BigQueryHybridSearchVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

store = BigQueryHybridSearchVectorStore(
    project_id="my-project",
    dataset_name="my_dataset",
    table_name="documents",
    location="US",
    embedding=VertexAIEmbeddings(model="gemini-embedding-001"),
    distance_type="COSINE",
    search_analyzer="LOG_ANALYZER",
)

# Pre-filter mode (default): keyword filter -> vector ranking
results = store.hybrid_search(
    query="How to optimize BigQuery performance?",
    text_query="BigQuery optimization",
    k=10,
)

# RRF mode: independent keyword + vector search -> merged ranking
results = store.hybrid_search_with_score(
    query="How to optimize BigQuery performance?",
    text_query="BigQuery optimization",
    k=10,
    fetch_k=50,
    hybrid_search_mode="rrf",
)
```

See the full [Hybrid Search](samples/hybrid_search.ipynb) tutorial.

## Related Projects

### Upstream Libraries

- [langchain-bigquery-graph](https://github.com/ksmin23/langchain-bigquery-graph) -- Standalone `GraphStore` and Graph RAG retrievers for BigQuery (the upstream of this package's graph features)
- [langchain-bigquery-hybridsearch](https://github.com/ksmin23/langchain-bigquery-hybridsearch) -- Standalone hybrid (vector + full-text) search vector store for BigQuery (the upstream of this package's hybrid search feature)

### Sample Applications

- [graph-rag-with-bigquery](https://github.com/ksmin23/my-adk-python-samples/tree/main/Graph-RAG/graph-rag-with-bigquery) -- Sample Agentic Graph RAG built with the Agent Development Kit (ADK), using BigQuery property graphs and this package's graph retrievers
- [rag-with-bigquery-hybridsearch](https://github.com/ksmin23/my-adk-python-samples/tree/main/RAG/rag-with-bigquery-hybridsearch) -- Sample Agentic RAG built with the Agent Development Kit (ADK), using this package's `BigQueryHybridSearchVectorStore` with Reciprocal Rank Fusion (RRF) to combine `VECTOR_SEARCH()` and `SEARCH()`

## License

MIT License. See [LICENSE](LICENSE) for details.
