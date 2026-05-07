GQL_EXAMPLES = """
The following query in backtick matches all persons in the graph FinGraph
whose birthday is before 1990-01-10 and
returns their name and birthday.
```
GRAPH FinGraph
MATCH (p:Person WHERE p.birthday < '1990-01-10')
RETURN p.name as name, p.birthday as birthday;
```

The following query in backtick finds the owner of the account with the most
incoming transfers by chaining multiple graph linear statements together.
```
GRAPH FinGraph
MATCH (:Account)-[:Transfers]->(account:Account)
RETURN account, COUNT(*) AS num_incoming_transfers
GROUP BY account
ORDER BY num_incoming_transfers DESC
LIMIT 1

NEXT

MATCH (account:Account)<-[:Owns]-(owner:Person)
RETURN account.id AS account_id, owner.name AS owner_name, num_incoming_transfers;
```

The following query finds all the destination accounts one to three transfers
away from a source Account with id equal to 7.
```
GRAPH FinGraph
MATCH (src:Account {{id: 7}})-[e:Transfers]->{{1, 3}}(dst:Account)
RETURN src.id AS src_account_id, dst.id AS dst_account_id;
```
Carefully note the syntax in the example above for path quantification,
that it is `[e:Transfers]->{{1, 3}}` and NOT `[e:Transfers*1..3]->`
"""

DEFAULT_GQL_TEMPLATE_PART0 = """
Create a BigQuery Graph GQL query for the question using the schema.
{gql_examples}
"""

DEFAULT_GQL_TEMPLATE_PART1 = """
Instructions:
Mention the name of the graph at the beginning.
Use only nodes and edge types, and properties included in the schema.
Do not use any node and edge type, or properties not included in the schema.
Always alias RETURN values.

Question: {question}
Schema: {schema}

Note:
Do not include any explanations or apologies.
Do not prefix query with `gql`
Do not include any backticks.
Start with GRAPH <graphname>
Output only the query statement.
Do not output any query that tries to modify or delete data.
"""

DEFAULT_GQL_TEMPLATE = (
    DEFAULT_GQL_TEMPLATE_PART0.format(gql_examples=GQL_EXAMPLES)
    + DEFAULT_GQL_TEMPLATE_PART1
)

BIGQUERY_GRAPH_QA_TEMPLATE = """
You are a helpful AI assistant.
Create a human readable answer for the question.
You should only use the information provided in the context and not use your internal knowledge.
Don't add any information.
Here is an example:

Question: Which funds own assets over 10M?
Context:[name:ABC Fund, name:Star fund]"
Helpful Answer: ABC Fund and Star fund have assets over 10M.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
You are given the following information:
- `Question`: the natural language question from the user
- `Graph Schema`: contains the schema of the graph database
- `Graph Query`: A BigQuery Graph GQL query equivalent of the question from the user used to extract context from the graph database
- `Context`: The response from the graph database as context
Information:
Question: {question}
Graph Schema: {graph_schema}
Graph Query: {graph_query}
Context: {context}

Helpful Answer:"""
