from pathlib import Path
import re
from .gen_structs import get_structs
from .gen_enums import get_enums
from .gen_stubs import get_stubs
from .utils import ptr_types, ptr_types_inv
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def latest_change(path: Path):
    return max(f.stat().st_mtime for f in path.rglob("*"))


def any_change():
    generated_dir = Path(__file__).parent
    template_dir = Path(__file__).parents[1] / "templates"
    pycache_dir = Path(__file__).parents[1] / "__pycache__"

    if latest_change(generated_dir, template_dir) > latest_change(pycache_dir):
        return True
    return False


def generate_bindings():
    template_dir = Path(__file__).parent / "templates"

    generated_dir = Path(__file__).parents[1] / "generated"
    generated_dir.mkdir(exist_ok=True)
    generated_dir.joinpath("__init__.py").touch()

    optix_types = Path("mylib/include/optix/optix_types.h")
    stubs = Path("mylib/include/optix/optix_stubs.h")
    my_types = Path("mylib/src/my_types.h")

    structs = get_structs(optix_types.read_text()) + get_structs(my_types.read_text())
    enums = get_enums(optix_types.read_text())
    stubs = get_stubs(stubs.read_text())

    types = set()
    types.update([f.type for s in structs for f in s.fields])
    types.update([a.type for s in stubs for a in s.args])

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

    params = dict(structs=structs, enums=enums, stubs=stubs)
    for thing in ["structs", "enums", "stubs"]:
        for ftype in ["cpp", "py"]:
            fname = f"{'c_' if ftype=='cpp' else ''}{thing}.{ftype}.jinja"
            out_text = env.get_template(fname).render(params)
            out_path = generated_dir.joinpath(fname.replace(".jinja", ""))
            if not out_path.exists() or out_path.read_text() != out_text:
                out_path.write_text(out_text)
                print(f"Generated {out_path}")
