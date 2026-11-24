"""Historical import for grace.paper_figures."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.paper_figures")
