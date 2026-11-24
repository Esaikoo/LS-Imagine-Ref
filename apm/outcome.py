"""Historical import for grace.outcome."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.outcome")
