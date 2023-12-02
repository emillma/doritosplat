import re
from pathlib import Path
from dataclasses import dataclass, field

flags = re.DOTALL | re.MULTILINE


@dataclass
class Value:
    name: str
    value: int
    comment: str = ""


@dataclass
class Enum:
    name: str
    values: list[Value] = field(default_factory=list)
    comment: str = ""
    start: int = 0


def get_enums(text: str):
    enum_pat = re.compile(r"^typedef enum (\w+)\n?{(.*?)\n} \1;", flags)
    enums = []
    for m in enum_pat.finditer(text):
        name = m[1]
        body = m[2]
        values = []
        for field_match in re.finditer(r"^ *(\w+)([^/]*?)(.*?)$", body, flags=flags):
            values.append(Value(*field_match.groups()))
        enums.append(Enum(name, values, start=m.start()))
    return enums
