from pathlib import Path
from mylib.lib_loader import load_lib
from .generate.generate import generate_bindings


generate_bindings()

from . import c_bindings
from .generated import structs, enums, stubs
