"""Historical import for grace.goal_guidance."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.goal_guidance")
