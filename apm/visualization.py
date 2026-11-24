"""Historical import for grace.visualization."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.visualization")
