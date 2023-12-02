from pathlib import Path
from mylib.generate import generate_bindings
from mylib.lib_loader import load_lib


def last_change(path: Path):
    return max(f.stat().st_mtime for f in path.rglob("*"))


template_dir = Path(__file__).parent / "templates"
pycache_dir = Path(__file__).parent / "__pycache__"

# if not pycache_dir.exists() or last_change(template_dir) > last_change(pycache_dir):
# generate_bindings()
generate_bindings()

from .bindings import ctypes, cmylib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generated import types
else:
    types = ctypes
