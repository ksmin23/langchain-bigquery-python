from __future__ import annotations

import re


def fix_gql_syntax(query: str) -> str:
    """Fixes common GQL path quantification syntax errors.

    Converts Cypher-style path quantification to ISO GQL syntax.
    e.g. `[e:cites*1..8]->` becomes `[e:cites]->{1,8}`
    """
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\.\.(\d+)\]->", r"-[\1:\2]->{\3,\4}", query)
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\]->", r"-[\1:\2]->{\3}", query)
    query = re.sub(r"<-\[(.*?):(\w+)\*(\d+)\.\.(\d+)\]-", r"<-[\1:\2]-{\3,\4}", query)
    query = re.sub(r"<-\[(.*?):(\w+)\*(\d+)\]-", r"<-[\1:\2]-{\3}", query)
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\.\.(\d+)\]-", r"-[\1:\2]-{\3,\4}", query)
    query = re.sub(r"-\[(.*?):(\w+)\*(\d+)\]-", r"-[\1:\2]-{\3}", query)
    return query


def extract_gql(text: str) -> str:
    """Extract GQL query from text, stripping code fences if present."""
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    query = matches[0] if matches else text
    return fix_gql_syntax(query)
