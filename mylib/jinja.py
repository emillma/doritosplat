import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass, field

flags = re.DOTALL | re.MULTILINE


@dataclass
class Field:
    type: str
    name: str


@dataclass
class Struct:
    name: str
    fields: list[Field] = field(default_factory=list)
    comment: str = ""


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


def get_structs(path: Path, filt=None):
    filt = filt or []
    text = Path(path).read_text()
    struc_pat = re.compile(r"^typedef struct (\w+)\n?{\n?(.*?)\n} \1;", flags)
    structs = []

    for m in filter(
        lambda m: not filt or m[1] in filt,
        struc_pat.finditer(text),
    ):
        name = m[1]
        body = m[2]
        fields = []
        for field_match in re.finditer(r"^ *((?:\w| )+) (\w+);", body, flags=flags):
            fields.append(Field(*field_match.groups()))

        structs.append(Struct(name, fields))
    return structs


def get_enums(path: Path):
    text = Path(path).read_text()
    enum_pat = re.compile(r"^typedef enum (\w+)\n?{(.*?)\n} \1;", flags)
    enums = []
    for m in enum_pat.finditer(text):
        name = m[1]
        body = m[2]
        values = []
        for field_match in re.finditer(r"^ *(\w+)([^/]*?)(.*?)$", body, flags=flags):
            values.append(Value(*field_match.groups()))
        enums.append(Enum(name, values))
    return enums


def generate_bindings():
    template_dir = Path(__file__).parent / "templates"
    generated_dir = Path(__file__).parent / "generated"
    optix_types = Path("mylib/include/optix/optix_types.h")
    mytypes = Path("mylib/src/mytypes.h")

    generated_dir.mkdir(exist_ok=True)
    generated_dir.joinpath("__init__.py").touch()

    structs = get_structs(mytypes)
    enums = get_enums(optix_types)

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    for fname in ["type_bindings.cpp.jinja", "types.py.jinja"]:
        out = env.get_template(fname).render(structs=structs, enums=enums)
        generated_dir.joinpath(fname.rpartition(".")[0]).write_text(out)
