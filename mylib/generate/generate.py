from pathlib import Path
from .gen_structs import get_structs
from .gen_enums import get_enums

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def latest_change(path: Path):
    return max(f.stat().st_mtime for f in path.rglob("*"))


def any_change():
    generate_dir = Path(__file__).parent
    generated_dir = Path(__file__).parents[1] / "generated"
    pycache_dir = Path(__file__).parents[1] / "__pycache__"

    if (
        latest_change(generate_dir) > latest_change(pycache_dir)
        or not generated_dir.is_dir()
    ):
        return True
    return False


def generate_bindings():
    if not any_change():
        return
    template_dir = Path(__file__).parent / "templates"
    generated_dir = Path(__file__).parents[1] / "generated"
    generated_dir.mkdir(exist_ok=True)
    generated_dir.joinpath("__init__.py").touch()
    bindings_h = generated_dir.joinpath("generated_bindings.h")
    bindings_h.write_text(
        "\n".join(
            (
                '#include "torch/extension.h"',
                "void bind_structs(py::module_ &m);",
                "void bind_enums(py::module_ &m);",
            )
        )
    )
    optix_types = Path("mylib/include/optix/optix_types.h")
    my_types = Path("/workspaces/doritosplat/mylib/src/my_structs.h")

    # my_types = Path("mylib/src/my_types.h")

    enums = get_enums(optix_types.read_text())
    enum_names = [e.name for e in enums]
    optix_structs = get_structs(optix_types.read_text(), enum_names)
    my_structs = get_structs(
        my_types.read_text(), enum_names + [s.name for s in optix_structs]
    )
    structs = optix_structs + my_structs

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

    render_params = [
        ["structs.cpp.jinja", "structs.cpp", dict(structs=structs)],
        ["structs.py.jinja", "structs.py", dict(structs=structs)],
        ["enums.cpp.jinja", "enums.cpp", dict(enums=enums)],
        ["enums.py.jinja", "enums.py", dict(enums=enums)],
    ]

    for template, out_name, params in render_params:
        out_text = env.get_template(template).render(params)
        out_path = generated_dir / out_name
        if not out_path.exists() or out_path.read_text() != out_text:
            out_path.write_text(out_text)
            print(f"Generated {out_path}")
