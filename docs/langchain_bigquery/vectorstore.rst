Hybrid Search Vector Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: langchain_bigquery.vectorstore
  :members:
  :private-members:
  :noindex:

BigQueryHybridSearchVectorStore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combines BigQuery ``VECTOR_SEARCH()`` (semantic similarity) with ``SEARCH()``
(full-text keyword matching) into a single retrieval step.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ``search_fields``
     - ``List[str]``
     - ``[content_field]``
     - Columns for ``SEARCH()``
   * - ``search_analyzer``
     - ``str``
     - ``LOG_ANALYZER``
     - Text analyzer
   * - ``search_analyzer_options``
     - ``str``
     - ``None``
     - Analyzer options (JSON)
   * - ``hybrid_search_mode``
     - ``str``
     - ``pre_filter``
     - Default mode (``pre_filter`` or ``rrf``)
   * - ``rrf_k``
     - ``int``
     - ``60``
     - RRF constant
   * - ``query_task_type``
     - ``Optional[str]``
     - ``None``
     - Task type for query embeddings
   * - ``document_task_type``
     - ``Optional[str]``
     - ``None``
     - Task type for document embeddings

All parameters from ``BigQueryVectorStore`` (``distance_type``, ``extra_fields``, etc.) are also supported.

Search Modes
"""""""""""""

**Pre-filter**

Uses ``SEARCH()`` to narrow candidates, then ``VECTOR_SEARCH()`` to rank by embedding distance.
Results **must** contain the keyword tokens.

::

   Query -> SEARCH(content, keywords) -> filtered rows -> VECTOR_SEARCH() -> top-k

**RRF (Reciprocal Rank Fusion)**

Runs both searches independently and combines results:

::

   Query -> VECTOR_SEARCH() -> vector_rank -+
                                            +-> RRF score -> top-k
   Query -> SEARCH()         -> text_rank  -+

RRF score: ``1/(k + vector_rank) + 1/(k + text_rank)`` where ``k`` defaults to 60.

Embedding Task Types
"""""""""""""""""""""

Vertex AI / Google Generative AI embedding models accept a
`task type <https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types>`_
hint that tunes the embedding for a specific downstream use case.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task type
     - Typical use
   * - ``RETRIEVAL_QUERY`` / ``RETRIEVAL_DOCUMENT``
     - Asymmetric retrieval (default for Google embeddings)
   * - ``QUESTION_ANSWERING``
     - Q&A retrieval -- set on **both** sides
   * - ``FACT_VERIFICATION``
     - Claim verification -- set on **both** sides
   * - ``SEMANTIC_SIMILARITY``
     - Symmetric similarity -- set on **both** sides
   * - ``CODE_RETRIEVAL_QUERY``
     - Code search -- set on the **query** side only

Score Semantics
""""""""""""""""

**Pre-filter mode** -- smaller is more similar (distance-based).

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - ``distance_type``
     - Distance returned
     - Interpretation
   * - ``EUCLIDEAN``
     - L2 norm
     - smaller = more similar
   * - ``COSINE``
     - ``1 - cos(theta)``
     - smaller = more similar
   * - ``DOT_PRODUCT``
     - ``-(a . b)`` (negated)
     - smaller = more similar

**RRF mode** -- larger is more relevant (fusion score-based).

BigQueryHybridSearchRetriever
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Custom retriever class returned by ``BigQueryHybridSearchVectorStore.as_retriever(search_type="hybrid")``.
Delegates to ``hybrid_search`` for the ``"hybrid"`` search type, and falls back to the
standard ``BigQueryVectorStore`` retriever for other search types.
