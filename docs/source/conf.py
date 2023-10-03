"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Nocturne"
copyright = "2023, The Nocturne Authors"
author = "The Nocturne Authors"

# The short X.Y version
version = "0.0.1"
# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_nb",  # See: https://myst-nb.readthedocs.io/en/latest/
    "sphinxcontrib.bibtex",  # See: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/quickstart.html
    "sphinx_autodoc_typehints",  # See: https://github.com/tox-dev/sphinx-autodoc-typehints
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.8.1/", None),
}

bibtex_bibfiles = ["references.bib"]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "path_to_docs": "docs",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_fullscreen_button": True,
    "use_issues_button": True,
    "use_source_button": True,
    "use_repository_button": True,
    "use_sidenotes": True,
    "repository_url": "https://github.com/emerge-lab/nocturne",
    "repository_branch": "main",
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "Nocturne GitHub",
            "url": "https://github.com/emerge-lab/nocturne",
            "icon": "fa-brands fa-github",
        },
    ],
}
html_static_path = ["_static"]
html_logo = "_static/logo.png"
# html_favicon = "_static/logo-square.svg"
html_title = "Nocturne"
html_copy_source = True

html_sidebars = {"**/**": ["sbt-sidebar-nav.html"]}

# -- Options for MySt-NB output -------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/index.html
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True

nb_execution_excludepatterns = [
    # Slow notebook
    # 'notebooks/Neural_Network_and_Data_Loading.*',
]

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Extension configuration -------------------------------------------------

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
always_document_param_types = True


# Tell sphinx autodoc how to render type aliases.
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "DTypeLike": "DTypeLike",
}
