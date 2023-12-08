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


src_dir = Path(__file__).parent / "src"
generated_dir = Path(__file__).parent / "generated"
mylib_dir = Path(__file__).parent

c_optix = load_lib(
    "c_optix",
    list(src_dir.glob("*.cpp")) + list(generated_dir.glob("*.cpp")),
)
# c_stubs = load_lib("c_stubs", [generated_dir / "c_stubs.cpp"])
# c_pointers = load_lib("c_pointers", [generated_dir / "c_pointers.cpp"])


# def create_context() -> "types.MyContext":
#     return c_optix.create_context()
