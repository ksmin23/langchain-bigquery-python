Graph Retrievers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: langchain_bigquery.graph_retriever
  :members:
  :private-members:
  :noindex:

BigQueryGraphVectorContextRetriever
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retriever that performs vector search on nodes in a ``BigQueryGraphStore``.
If ``expand_by_hops`` is provided, the nodes (and edges) at a distance up to
``expand_by_hops`` will also be returned.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 25 15 10 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``graph_store``
     - ``BigQueryGraphStore``
     - required
     - The graph store to search
   * - ``embedding_service``
     - ``Embeddings``
     - required
     - Embedding model for vectorizing queries
   * - ``label_expr``
     - ``str``
     - ``"%"``
     - Label expression to filter nodes
   * - ``return_properties_list``
     - ``List[str]``
     - ``[]``
     - Specific properties to return (mutually exclusive with ``expand_by_hops``)
   * - ``embeddings_column``
     - ``str``
     - ``"embedding"``
     - Column name storing node embeddings
   * - ``distance_strategy``
     - ``DistanceStrategy``
     - ``COSINE``
     - ``COSINE`` or ``EUCLIDEAN``
   * - ``top_k``
     - ``int``
     - ``3``
     - Number of vector similarity matches
   * - ``expand_by_hops``
     - ``int``
     - ``-1``
     - Hops to traverse for neighborhood expansion (mutually exclusive with ``return_properties_list``)
   * - ``k``
     - ``int``
     - ``10``
     - Max number of graph results to return

.. note::

   Exactly one of ``return_properties_list`` or ``expand_by_hops`` must be provided.
   With ``return_properties_list``, results come from a direct SQL query on the base table.
   With ``expand_by_hops``, vector search finds matching node IDs via SQL,
   then GQL traverses the graph neighborhood.

BigQueryGraphTextToGQLRetriever
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Translates natural language queries to GQL and executes them
against a ``BigQueryGraphStore``.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 20 25 10 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ``graph_store``
     - ``BigQueryGraphStore``
     - required
     - The graph store to query
   * - ``llm``
     - ``BaseLanguageModel``
     - required
     - LLM for GQL generation
   * - ``k``
     - ``int``
     - ``10``
     - Max number of results to return
   * - ``selector``
     - ``SemanticSimilarityExampleSelector``
     - ``None``
     - Few-shot example selector (auto-created via ``from_params``)

DistanceStrategy
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from langchain_bigquery import DistanceStrategy

   DistanceStrategy.COSINE      # COSINE_DISTANCE
   DistanceStrategy.EUCLIDEAN   # EUCLIDEAN_DISTANCE
