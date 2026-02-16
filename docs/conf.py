# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version
from inspect import cleandoc
from pathlib import Path

import git
import nbformat
import nbsphinx
from packaging.version import parse

sys.path.insert(0, os.path.abspath("../"))


os.environ["SPHINX"] = "True"

# -- Project information -----------------------------------------------------

project = "OpenFE Analysis"
copyright = "2026, The OpenFE Development Team"
author = "The OpenFE Development Team"
version = f"{parse(version('openfe_analysis')).major}.{parse(version('openfe_analysis')).minor}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click.ext",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_toolbox.collapse",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "docs._ext.sass",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
]
suppress_warnings = ["config.cache"]  # https://github.com/sphinx-doc/sphinx/issues/12300

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scikit.learn": ("https://scikit-learn.org/stable", None),
    "openmm": ("https://docs.openmm.org/latest/api-python/", None),
    "rdkit": ("https://www.rdkit.org/docs", None),
    "openeye": ("https://docs.eyesopen.com/toolkits/python/", None),
    "mdtraj": ("https://www.mdtraj.org/1.9.5/", None),
    "openff.units": ("https://docs.openforcefield.org/projects/units/en/stable", None),
    "gufe": ("https://gufe.openfree.energy/en/latest/", None),
}

autoclass_content = "both"
# Make sure labels are unique
# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html#confval-autosectionlabel_prefix_document
autosectionlabel_prefix_document = True

autodoc_pydantic_model_show_json = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": "GufeTokenizable,BaseModel",
    "undoc-members": True,
    "special-members": "__call__",
}
toc_object_entries_show_parents = "hide"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "**/Thumbs.db",
    "**/.DS_Store",
    "_ext",
    "_sass",
    "**/README.md",
    "ExampleNotebooks",
]

autodoc_mock_imports = [
    "MDAnalysis",
]

# Extensions for the myst parser
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "smartquotes",
    "replacements",
    "deflist",
    "attrs_inline",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "ofe_sphinx_theme"
html_theme_options = {
    "logo": {"text": "openfe-analysis docs"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/OpenFreeEnergy/openfe_analysis",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "accent_color": "cantina-purple",
    "navigation_with_keys": False,
}
html_logo = "_static/OFE-color-icon.svg"
html_favicon = "_static/OFE-color-icon.svg"
# temporary fix, see https://github.com/pydata/pydata-sphinx-theme/issues/1662
html_sidebars = {
    "installation": [],
    "CHANGELOG": [],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# replace macros
rst_prolog = """
.. |rdkit.mol| replace:: :class:`rdkit.Chem.rdchem.Mol`
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "css/custom-api.css",
    "css/deflist-flowchart.css",
]

# custom-api.css is compiled from custom-api.scss
sass_src_dir = "_sass"
sass_out_dir = "_static/css"
