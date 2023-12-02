from pathlib import Path
import re
from .structs import get_structs
from .enums import get_enums
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def latest_change(*paths: Path):
    newest = 0
    for path in paths:
        newest = max(newest, max(f.stat().st_mtime for f in path.rglob("*")))
    return newest


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
    mytypes = Path("mylib/src/mytypes.h")

    structs = get_structs(optix_types.read_text())
    enums = get_enums(optix_types.read_text())
    struct_filter = [
        "OptixAccelBuildOptions",
        "OptixMotionOptions",
        "OptixBuildInput",
        "OptixBuildInputTriangleArray",
        "OptixAccelBufferSizes",
    ]

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

    params = dict(structs=structs, enums=enums)
    for thing in ["structs", "enums"]:
        for ftype in ["cpp", "py"]:
            fname = f"{'c_' if ftype=='cpp' else ''}{thing}.{ftype}.jinja"
            out = env.get_template(fname).render(params)
            generated_dir.joinpath(fname.replace(".jinja", "")).write_text(out)
