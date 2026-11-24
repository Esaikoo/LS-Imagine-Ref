"""Historical import for grace.legacy.control."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.legacy.control")
