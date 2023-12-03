from pathlib import Path
from dataclasses import dataclass, field
import re
from .utils import ptr_typedefs
from .gen_stubs import Argument, Function


@dataclass
class Pointer:
    type: str
    name: str
    is_const: bool = False


flags = re.DOTALL | re.MULTILINE


def gettype(type: str):
    # a, b, c = type.rpartition("*")
    return ptr_typedefs.get(type, type)


def is_pointer(type: str):
    type = gettype(type)
    return "*" in type


def get_pointers(stubs: list[Function]):
    ptr_types = set()

    for stub in stubs:
        for arg in stub.args:
            type = gettype(arg.type)
            if is_pointer(type):
                a, b, c = type.rpartition("*")

                ptr_types.add(a + b)
    return ptr_types


def pytype(type):
    if is_pointer(type):
        type = gettype(type)
        a, b, c = type.rpartition("*")
        return f"ptr_wrapper<{a.strip()}>"
    return f"{type}"


def ptr_name(type):
    a, b, c = type.rpartition("*")
    return a.replace(" ", "_").replace("*", "ptr") + "_pyptr"


def c2py(type, name):
    if is_pointer(type):
        return f"{pytype(type)}({name})"
    return name


def py2c(type, name):
    if is_pointer(type):
        return f"{name}.get()"
    return name


def pyargs(args: list[Argument]):
    return ", ".join([f"{pytype(arg.type)} &{arg.name}" for arg in args])


def callargs(args: list[Argument]):
    return ", ".join([py2c(arg.type, arg.name) for arg in args])
