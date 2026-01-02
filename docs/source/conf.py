# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pyhanami'
copyright = '2025, Marta Alerany Solé, Kai Keller'
author = 'Marta Alerany Solé, Kai Keller'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "myst_parser",                  # Optional Markdown support
    "sphinx.ext.todo",              # Add todo directives
    "sphinx.ext.viewcode",          # Add links to highlighted source code
    "sphinx.ext.autodoc",           # Pull in docstrings
    "sphinx.ext.mathjax",           # Render math equations
    "sphinx_rtd_theme",             # Theme for Read the Docs
    "sphinx.ext.autosummary",       # Auto-generate summary tables
    "sphinx.ext.napoleon",          # Support Google/NumPy-style docstrings
]
templates_path = ['_templates']
exclude_patterns = []
myst_heading_anchors = 2

# MyST parser configuration to properly render math equations
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# Mock imports for packages that might not be available during doc building
autodoc_mock_imports = [
    'cartopy',
    'cftime',
    'cmocean',
    'dask',
    'eofs',
    'ESMF',
    'esmpy',
    'intake',
    'intake_esm',
    'matplotlib',
    'nc_time_axis',
    'netCDF4',
    'numpy',
    'pandas',
    'scipy',
    'seaborn',
    'sklearn',
    'statsmodels',
    'xarray',
    'xeofs',
    'xesmf',
]

# # Keep order of the members in the source python files
# autodoc_member_order = 'bysource'

# Add path to source code
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme' # Change theme to common black and blue Read the Docs format
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',   # Customize tables formatting
]