from pathlib import Path

from torch.utils import cpp_extension
import json
from subprocess import run
import tempfile
import sys

cwd = Path.cwd()
props = json.loads((cwd / ".vscode/c_cpp_properties.json").read_text())
config = next(c for c in props["configurations"] if c["name"] == "OptiX")
include_path: list[str] = config["includePath"]
include_dirs = [p.replace("${workspaceFolder}", str(cwd)) for p in include_path]


def load_lib(name: str, sources: Path):
    if not isinstance(sources, list):
        sources = [p for f in ["cu", "cpp"] for p in sources.glob(f"*.{f}")]

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True, parents=True)
    try:
        module = cpp_extension.load(
            name=name,
            sources=[str(s) for s in sources],
            build_directory=str(build_dir),
            verbose=True,
            keep_intermediates=True,
            extra_include_paths=[str(p) for p in include_dirs],
            with_cuda=True,
            extra_cuda_cflags=[
                "-arch=sm_89",
                "--use_fast_math",
                "--ptxas-options=-v",
            ],
        )
    except Exception as e:
        print("\n\n------PYCRASH------\n\n")
        raise e
    return module


def get_optixir(path: Path, debug=False):
    extra = (
        ["-G", "--generate-line-info", "debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_MODERATE"]
        if debug
        else []
    )
    with tempfile.NamedTemporaryFile(suffix=".optixir") as fp:
        ptx_args = [
            "nvcc",
            "-optix-ir",
            "--use_fast_math",
            "--machine=64",
            "--relocatable-device-code=true",
            "--keep-device-functions",
            *extra,
            *("-I" + str(p) for p in include_dirs),
            str(path),
            "-o",
            fp.name,
        ]

        run(ptx_args, check=True)
        data = fp.read()
    return data
