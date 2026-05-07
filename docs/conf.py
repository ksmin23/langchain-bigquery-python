# -*- coding: utf-8 -*-
#
# langchain-bigquery documentation build configuration file

import os
import shlex
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

__version__ = ""

needs_sphinx = "1.5.5"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "recommonmark",
]

autoclass_content = "both"
autodoc_default_options = {"members": True}
autosummary_generate = True

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
root_doc = "index"

project = "langchain-bigquery"
copyright = "2024"
author = "Unknown"

release = __version__
version = ".".join(release.split(".")[0:2])

language = None

exclude_patterns = [
    "_build",
]

pygments_style = "sphinx"
todo_include_todos = True

html_theme = "alabaster"
html_theme_options = {
    "description": "LangChain integrations for Google Cloud BigQuery",
    "github_user": "ksmin23",
    "github_repo": "langchain-bigquery-python",
    "github_banner": True,
    "font_family": "'Roboto', Georgia, sans",
    "head_font_family": "'Roboto', Georgia, serif",
    "code_font_family": "'Roboto Mono', 'Consolas', monospace",
}

html_static_path = ["_static"]
htmlhelp_basename = "langchain-bigquery-doc"

suppress_warnings = ["ref.python"]

latex_elements = {}
latex_documents = [
    (
        root_doc,
        "langchain-bigquery.tex",
        "langchain-bigquery Documentation",
        author,
        "manual",
    )
]

man_pages = [
    (
        root_doc,
        "langchain-bigquery",
        "langchain-bigquery Documentation",
        [author],
        1,
    )
]

texinfo_documents = [
    (
        root_doc,
        "langchain-bigquery",
        "langchain-bigquery Documentation",
        author,
        "langchain-bigquery",
        "langchain-bigquery Library",
        "APIs",
    )
]

intersphinx_mapping = {
    "python": ("https://python.readthedocs.org/en/latest/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
