# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme

project = 'IAMReX'
copyright = '2025, IAMReX'
author = 'YD Z'
release = '0.1.0'

bibtex_bibfiles = ["refs.bib"]
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
main_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.mathjax',
              'sphinxcontrib.bibtex',
              'sphinx.ext.githubpages',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinx.ext.autosectionlabel'
              ]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
