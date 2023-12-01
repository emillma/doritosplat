from typing import TYPE_CHECKING
from pathlib import Path
from mylib.generate import generate_bindings
from mylib.lib_loader import load_lib
import torch
import re

template_dir = Path(__file__).parent / "templates"
pycache_dir = Path(__file__).parent / "__pycache__"


def last_change(path: Path):
    return max(f.stat().st_mtime for f in path.rglob("*"))


# if not pycache_dir.exists() or last_change(template_dir) > last_change(pycache_dir):
#     generate_bindings()
# generate_bindings()


def sources(path: Path):
    out = []
    for f in path.rglob("*"):
        if f.suffix in [".cu", ".cpp"]:
            out.append(f)
    return out


mylib_dir = Path(__file__).parent

ctypes = load_lib("types", sources(mylib_dir / "generated"))
cmylib = load_lib("cmylib", mylib_dir / "src")

if TYPE_CHECKING:
    from .generated import types
else:
    types = ctypes
