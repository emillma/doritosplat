from pathlib import Path
from dataclasses import dataclass, field
import re

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
    "std::vector<.*>",
    "std::array<.*>",
    "std::tuple<.*>",
    "std::string",
    "torch::Tensor",
    "CUdeviceptr",
    "OptixTraversableHandle",
]


@dataclass
class Field:
    type: str
    name: str


@dataclass
class Struct:
    name: str
    fields: list[Field] = field(default_factory=list)
    comment: str = ""
    start: int = 0


def get_structs(text: str, estra_types: list[str] = []):
    struc_pat = re.compile(r"^typedef struct[^{;]*?{\n?([^}]*?)\n} (\w+);", flags)
    field_pat = re.compile(r"^ *([a-zA-Z0-9 <>:_\*]+?)(\w+);", flags)
    structs = []

    struct_matches = dict()
    for m in struc_pat.finditer(text):
        name = m[2]
        if name in ignore_structs:
            continue
        struct_matches[name] = m

    ok = available_types + estra_types + list(struct_matches.keys())
    ok_pat = [re.compile(f"^{t}$") for t in ok]

    for name, m in struct_matches.items():
        name = m[2]
        body = m[1]
        fields = []
        for field_match in field_pat.finditer(body):
            field_type = field_match[1].strip()
            field_name = field_match[2]
            field = Field(field_type, field_name)
            if any(pat.match(field_type) for pat in ok_pat):
                fields.append(field)

        structs.append(Struct(name, fields, start=m.start()))
    return structs
