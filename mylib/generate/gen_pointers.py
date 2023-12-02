from pathlib import Path
from dataclasses import dataclass, field
import re
from .utils import ptr_types, ignore_types


@dataclass
class Pointer:
    type: str
    name: str
    py_type: str = field(init=False)
    is_const: bool = False

    def __post_init__(self):
        if self.type in pointer_types or self.type.endswith("*"):
            self.py_type = f"size_t"
        else:
            self.py_type = self.type


flags = re.DOTALL | re.MULTILINE


def get_pointers():
    pointers = []
    # for k, v in ptr_types.items():
