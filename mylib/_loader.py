from pathlib import Path

from torch.utils import cpp_extension
import jsonc
from subprocess import run

cwd = Path.cwd()
this_dir = Path(__file__).parent

props = jsonc.loads((cwd / ".vscode/c_cpp_properties.json").read_text())
config = next(c for c in props["configurations"] if c["name"] == "OptiX")
include_path: list[str] = config["includePath"]
include_dirs = [p.replace("${workspaceFolder}", str(cwd)) for p in include_path]

build_dir = this_dir / "build"
build_dir.mkdir(exist_ok=True)


def load(name: str, source_dir: Path, with_cuda=True):
    try:
        module = cpp_extension.load(
            name=name,
            sources=[str(p) for f in ["cu", "cpp"] for p in source_dir.glob(f"*.{f}")],
            build_directory=str(build_dir),
            verbose=True,
            extra_include_paths=[str(p) for p in include_dirs],
            with_cuda=with_cuda,
            extra_cuda_cflags=[
                "-arch=sm_89",
                "--use_fast_math",
                "--ptxas-options=-v",
            ],
        )
    except Exception as e:
        print(e)
        raise e
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
