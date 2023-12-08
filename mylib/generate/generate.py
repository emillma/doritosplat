from pathlib import Path
import re
from .gen_structs import get_structs
from .gen_enums import get_enums

# from .gen_pointers import get_pointers, c2py, py2c, pyargs, callargs, pytype, ptr_name
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
    bindings_h = generated_dir.joinpath("generated_bindings.h")
    bindings_h.write_text(
        "\n".join(
            (
                '#include "torch/extension.h"',
                "void bind_enums(py::module_ &m);",
                "void bind_structs(py::module_ &m);",
            )
        )
    )
    optix_types = Path("mylib/include/optix/optix_types.h")
    # my_types = Path("mylib/src/my_types.h")

    structs = get_structs(optix_types.read_text())
    enums = get_enums(optix_types.read_text())

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

    params = dict(
        structs=structs,
        enums=enums,
    )
    for thing in ["structs", "enums"]:
        for ftype in ["cpp", "py"]:
            fname = f"{thing}.{ftype}.jinja"
            out_text = env.get_template(fname).render(params)
            out_path = generated_dir.joinpath(fname.replace(".jinja", ""))
            if not out_path.exists() or out_path.read_text() != out_text:
                out_path.write_text(out_text)
                print(f"Generated {out_path}")
