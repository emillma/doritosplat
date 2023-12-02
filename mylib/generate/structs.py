from pathlib import Path
from dataclasses import dataclass, field
import re
from .utils import pointer_types, ignore_types

flags = re.DOTALL | re.MULTILINE

ignore_structs = [
    "OptixInvalidRayExceptionDetails",
    "OptixParameterMismatchExceptionDetails",
]


@dataclass
class Field:
    type: str
    name: str
    py_type: str = field(init=False)
    is_const: bool = False

    def __post_init__(self):
        if self.type in pointer_types or self.type.endswith("*"):
            self.py_type = f"size_t"
        else:
            self.py_type = self.type


@dataclass
class Struct:
    name: str
    fields: list[Field] = field(default_factory=list)
    comment: str = ""
    start: int = 0


def get_structs(text: str):
    struc_pat = re.compile(r"^typedef struct (\w+)\n?{\n?(.*?)\n} \1;", flags)
    field_pat = re.compile(r"^ *([a-zA-Z0-9 _\*]+?)(\w+);", flags)
    structs = []

    for m in struc_pat.finditer(text):
        name = m[1]
        body = m[2]
        if name in ignore_structs:
            continue
        fields = []
        for field_match in field_pat.finditer(body):
            field_type = field_match[1].strip()
            field_name = field_match[2]
            if field_type in ignore_types:
                continue
            fields.append(Field(field_type, field_name))

        structs.append(Struct(name, fields, start=m.start()))
    return structs
