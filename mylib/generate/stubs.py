from dataclasses import dataclass, field
import re
from .utils import pointer_types

flags = re.DOTALL | re.MULTILINE


@dataclass
class Argument:
    type: str
    name: str
    py_type: str = field(init=False)

    def __post_init__(self):
        if self.type in pointer_types or self.type.endswith("*"):
            self.py_type = f"size_t"
        else:
            self.py_type = self.type


@dataclass
class Function:
    name: str
    args: list[Argument]
    return_type: str

    def c_py_arg_string(self):
        return ", ".join([f"{arg.py_type} {arg.name}" for arg in self.args])

    def c_py2c_arg_string(self):
        return ", ".join(
            [
                f"({arg.type}){arg.name}" if arg.type != arg.py_type else arg.name
                for arg in self.args
            ]
        )

    def c_tuple_input(self):
        return ", ".join([f"{arg.name}" for arg in self.args])

    def py_args(self):
        return ", ".join([f"{arg.name}" for arg in self.args])

    def py_args_annotated(self):
        return ", ".join(
            [
                f'{arg.name}:"{arg.type}{f"({arg.py_type})" if arg.type != arg.py_type else ""}"'
                for arg in self.args
            ]
        )


def get_stubs(text: str):
    stub_pat = re.compile(r"^ *inline OptixResult (\w+)\((.*?)\)", flags)
    arg_pat = re.compile(r"([a-zA-Z0-9 _\*]+?)(\w+)$")
    functions = []
    for stub in stub_pat.finditer(text):
        name = stub[1]
        arg_string = stub[2]
        args = []
        for arg in arg_string.replace("\n", "").split(","):
            if arg == "void":
                continue
            m = arg_pat.match(arg)
            arg_type = m[1].strip()
            arg_name = m[2].strip()
            args.append(Argument(arg_type, arg_name))
        functions.append(Function(name, args, "OptixResult"))
    return functions
