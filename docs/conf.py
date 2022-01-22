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
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../pyjjasim/'))

print(sys.path)
# -- Project information -----------------------------------------------------

project = 'pyjjasim'
copyright = '2022, MartijnLankhorst'
author = 'MartijnLankhorst'

# The full version, including alpha/beta/rc tags
release = '2.2.7'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.mathjax',
#     'numpydoc'
# ]
# extensions = [
#     'numpydoc'
# ]

# extensions = [
#     'sphinx.ext.autodoc',
#     'numpydoc',
#     'sphinx.ext.intersphinx',
#     'sphinx.ext.coverage',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.graphviz',
#     'sphinx.ext.ifconfig',
#     'matplotlib.sphinxext.plot_directive',
#     'IPython.sphinxext.ipython_console_highlighting',
#     'IPython.sphinxext.ipython_directive',
#     'sphinx.ext.mathjax',
#     'sphinx_panels',
# ]
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.viewcode',
]
# autodoc_default_flags = ['members', 'undoc-members']

# autodoc_default_options = {
#     'members': 'var1, var2',
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': False,
#     'exclude-members': '__weakref__'
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


master_doc = 'index'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_use_modindex = True

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }