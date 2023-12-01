from pathlib import Path

from torch.utils import cpp_extension
import jsonc
from subprocess import run

cwd = Path.cwd()
props = jsonc.loads((cwd / ".vscode/c_cpp_properties.json").read_text())
config = next(c for c in props["configurations"] if c["name"] == "OptiX")
include_path: list[str] = config["includePath"]
include_dirs = [p.replace("${workspaceFolder}", str(cwd)) for p in include_path]


def load_lib(name: str, sources: Path):
    if not isinstance(sources, list):
        sources = [p for f in ["cu", "cpp"] for p in sources.glob(f"*.{f}")]

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True, parents=True)
    module = cpp_extension.load(
        name=name,
        sources=[str(s) for s in sources],
        build_directory=str(build_dir),
        verbose=True,
        extra_include_paths=[str(p) for p in include_dirs],
        with_cuda=True,
        extra_cuda_cflags=[
            "-arch=sm_89",
            "--use_fast_math",
            "--ptxas-options=-v",
        ],
    )

    return module


def ptx(path: Path, out: Path):
    out.parent.mkdir(exist_ok=True, parents=True)
    ptx_args = [
        "nvcc",
        # "--cudart=static",
        "--ptx",
        *("-I" + str(p) for p in include_dirs),
        "-o",
        str(out),
        str(path),
    ]
    run(ptx_args, check=True)
