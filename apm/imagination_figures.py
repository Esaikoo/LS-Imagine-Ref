"""Historical import for grace.imagination_figures."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.imagination_figures")
