"""Historical import for grace.goal_reference."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.goal_reference")
