Graph Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: langchain_bigquery.graph_store
  :members:
  :private-members:
  :noindex:

BigQueryGraphStore
^^^^^^^^^^^^^^^^^^

.. list-table:: Parameters
   :header-rows: 1
   :widths: 20 15 10 55

   * - Parameter
     - Type
     - Default
     - Description
   * - ``project_id``
     - ``str``
     - required
     - Google Cloud project ID
   * - ``dataset_id``
     - ``str``
     - required
     - BigQuery dataset ID
   * - ``graph_name``
     - ``str``
     - required
     - Property graph name
   * - ``client``
     - ``bigquery.Client``
     - ``None``
     - Optional pre-configured client
   * - ``location``
     - ``str``
     - ``None``
     - BigQuery location (e.g., ``us-central1``). Ignored if ``client`` is provided
   * - ``use_flexible_schema``
     - ``bool``
     - ``False``
     - Use JSON-based flexible schema
   * - ``static_node_properties``
     - ``List[str]``
     - ``[]``
     - Properties stored as static columns in flexible schema
   * - ``static_edge_properties``
     - ``List[str]``
     - ``[]``
     - Properties stored as static columns in flexible schema

.. list-table:: Methods
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``query(query, params)``
     - Execute a GQL query and return results
   * - ``get_schema``
     - Property graph schema as JSON string
   * - ``get_structured_schema``
     - Schema as a Python dictionary
   * - ``get_ddl()``
     - Property graph DDL as string
   * - ``add_graph_documents(docs)``
     - Create tables, graph DDL, and insert data
   * - ``refresh_schema()``
     - Reload schema from ``INFORMATION_SCHEMA``
   * - ``cleanup()``
     - Drop property graph and all associated tables
