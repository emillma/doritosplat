from pathlib import Path
from pycparser import parse_file, CParser
from subprocess import run

parser = CParser()
cpp_file = Path(__file__).parent / "optix_types.cpp"
cmd = ["nvcc", "--preprocess", str(cpp_file)]
out = run(cmd, capture_output=True, check=True)
ast = parser.parse(out.stdout.decode())
print(out)
