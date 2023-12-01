from pathlib import Path
from mylib._loader import load
from mylib import structs

cpp = load("funcs", Path(__file__).parent, False)


def foo(
    a: structs.OptixDisplacementMicromapDesc,
) -> structs.OptixDisplacementMicromapDesc:
    return cpp.foo(a)
