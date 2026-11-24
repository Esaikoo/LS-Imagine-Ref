"""Historical import for grace.control."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.control")
