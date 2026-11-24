"""Historical import for grace.odometry."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.odometry")
