from pathlib import Path
import shutil
from .generate.generate import generate_bindings
from .lib_loader import load_lib

FORCE = True

mylib_dir = Path(__file__).parent
src_dir = mylib_dir / "src"
generate_dir = mylib_dir / "generate"
generated_dir = mylib_dir / "generated"
build_dir = mylib_dir / "build"

if FORCE:
    shutil.rmtree(generated_dir, ignore_errors=True)
    shutil.rmtree(build_dir, ignore_errors=True)


def latest_change(*dirs: Path):
    return max((f.stat().st_mtime for d in dirs for f in d.rglob("*")), default=0)


if FORCE or latest_change(generate_dir) > latest_change(generated_dir):
    generate_bindings()


module_name = "_clib"
if latest_change(src_dir, generated_dir) > latest_change(build_dir):
    _clib = load_lib(
        module_name,
        [
            *src_dir.glob("*.cpp"),
            *generated_dir.glob("*.cpp"),
        ],
    )
else:
    import importlib.util
    import importlib.abc

    spec = importlib.util.spec_from_file_location(
        module_name, build_dir / f"{module_name}.so"
    )
    assert spec is not None
    _clib = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(_clib)
