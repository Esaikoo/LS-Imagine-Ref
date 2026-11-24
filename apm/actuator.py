"""Historical import for grace.actuator."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("grace.actuator")
