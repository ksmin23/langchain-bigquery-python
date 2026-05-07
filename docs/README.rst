BigQuery for LangChain
=================================

|pypi| |versions|

- `Product Documentation`_

.. |pypi| image:: https://img.shields.io/pypi/v/langchain-bigquery.svg
   :target: https://pypi.org/project/langchain-bigquery/
.. |versions| image:: https://img.shields.io/pypi/pyversions/langchain-bigquery.svg
   :target: https://pypi.org/project/langchain-bigquery/
.. _Product Documentation: https://cloud.google.com/bigquery

Quick Start
-----------

In order to use this library, you first need to go through the following
steps:

1. `Select or create a Cloud Platform project.`_
2. `Enable billing for your project.`_
3. `Enable the Google Cloud BigQuery API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the Google Cloud BigQuery API.: https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com
.. _Setup Authentication.: https://googleapis.dev/python/google-api-core/latest/auth.html

Installation
~~~~~~~~~~~~

Install this library in a `virtualenv`_ using pip.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

.. code-block:: console

   pip install langchain-bigquery

BigQuery Graph Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``BigQueryGraphStore`` for storing and querying property graphs in BigQuery.

.. code-block:: python

    from langchain_bigquery import BigQueryGraphStore

    graph = BigQueryGraphStore(
        project_id="my-project",
        dataset_name="my_dataset",
        graph_name="my_graph",
    )

BigQuery Hybrid Search Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``BigQueryHybridSearchVectorStore`` for hybrid (vector + full-text) search.

.. code-block:: python

    from langchain_bigquery import BigQueryHybridSearchVectorStore

License
-------

MIT License. See `LICENSE <../LICENSE>`_ for details.
