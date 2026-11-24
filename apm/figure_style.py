"""Historical import for grace.figure_style."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.figure_style")
