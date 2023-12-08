from pathlib import Path
from dataclasses import dataclass, field
import re
from .utils import pointer_types, ignore_types

flags = re.DOTALL | re.MULTILINE

ignore_structs = [
    "OptixInvalidRayExceptionDetails",
    "OptixParameterMismatchExceptionDetails",
]
available_types = [
    "char",
    "short",
    "int",
    "long",
    "unsigned char",
    "unsigned short",
    "unsigned int",
    "unsigned long",
    "size_t",
    "float",
    "double",
    "CUdeviceptr",
]


@dataclass
class Field:
    type: str
    name: str
    is_const: bool = False
    available = False

    def __post_init__(self):
        self.available = self.type in available_types


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

    struct_matches = dict()
    for m in struc_pat.finditer(text):
        name = m[1]
        if name in ignore_structs:
            continue
        struct_matches[name] = m

    for name, m in struct_matches.items():
        name = m[1]
        body = m[2]
        fields = []
        for field_match in field_pat.finditer(body):
            field_type = field_match[1].strip()
            field_name = field_match[2]
            if field_type in ignore_types:
                continue
            fields.append(Field(field_type, field_name))

        structs.append(Struct(name, fields, start=m.start()))
    return structs
