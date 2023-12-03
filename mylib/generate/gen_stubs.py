from dataclasses import dataclass, field
import re
from .utils import pointer_types, ignore_types

flags = re.DOTALL | re.MULTILINE


@dataclass
class Argument:
    type: str
    name: str


@dataclass
class Function:
    name: str
    args: list[Argument]
    return_type: str


def get_stubs(text: str):
    stub_pat = re.compile(r"^ *inline OptixResult (\w+)\((.*?)\)", flags)
    arg_pat = re.compile(r"([a-zA-Z0-9 _\*]+?)(\w+)$")
    functions = []
    for stub in stub_pat.finditer(text):
        name = stub[1]
        arg_string = stub[2]
        args = []
        skip = False
        for arg in arg_string.replace("\n", "").split(","):
            if arg == "void":
                continue
            m = arg_pat.match(arg)
            arg_type = m[1].strip()
            arg_name = m[2].strip()
            if arg_type in ignore_types:
                skip = True
                break
            args.append(Argument(arg_type, arg_name))
        if skip:
            continue
        functions.append(Function(name, args, "OptixResult"))
    return functions
