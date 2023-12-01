from pathlib import Path
from ._generate import generate_bindings

generate_bindings()
from .generated.structs import *
