"""Historical import for grace.diagnostics."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.diagnostics")
