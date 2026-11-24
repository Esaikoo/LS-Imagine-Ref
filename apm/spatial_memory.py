"""Historical import for grace.spatial_memory."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.spatial_memory")
