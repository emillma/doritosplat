from pathlib import Path
from mylib.lib_loader import load_lib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generated import types


def sources(path: Path):
    out = []
    for f in path.rglob("*"):
        if f.suffix in [".cu", ".cpp"]:
            out.append(f)
    return out


mylib_dir = Path(__file__).parent
ctypes = load_lib("types", sources(mylib_dir / "generated"))
cmylib = load_lib("cmylib", mylib_dir / "src")


def create_context() -> "types.MyContext":
    return cmylib.create_context()
