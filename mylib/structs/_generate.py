import re
from pathlib import Path
from pycparser import c_parser, c_ast, parse_file
from jinja2 import Template, Environment, FileSystemLoader
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


def parse_struct(match: re.Match):
    name = match[1]
    body = match[2]
    fields = []
    for field_match in re.finditer(r"^ *((?:\w| )+) (\w+);", body, flags=flags):
        fields.append(Field(*field_match.groups()))

    return Struct(name, fields)


def parse_enum(match: re.Match):
    name = match[1]
    body = match[2]
    fields = []
    for field_match in re.finditer(r"^ *((?:\w| )+) (\w+);", body, flags=flags):
        fields.append(Field(*field_match.groups()))

    return Struct(name, fields)


def get_enums(text: str):
    enum_pat = re.compile(r"typedef enum (\w+)\n?{(.*?)\n} \1;", flags)
    enums = []
    for enum_match in enum_pat.finditer(text):
        enums.append(parse_enum(enum_match))
    return enums


def get_structs(text: str):
    struc_pat = re.compile(r"typedef struct (\w+)\n?{\n?(.*?)\n} \1;", flags)
    structs = []
    for struct_match in struc_pat.finditer(text):
        if struct_match[1] != "OptixDisplacementMicromapDesc":
            continue
        structs.append(parse_struct(struct_match))
    return structs


def generate_bindings():
    template_dir = Path("mylib/structs/templates")
    generated_dir = Path("mylib/structs/generated")
    generated_dir.mkdir(exist_ok=True)
    generated_dir.joinpath("__init__.py").touch()
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    types_in = Path("mylib/include/optix/optix_types.h")
    structs = get_structs(types_in.read_text())

    for fname in ["bindings.cpp.jinja", "structs.py.jinja"]:
        template = env.get_template(fname)
        out = template.render(structs=structs)
        generated_dir.joinpath(template.name.rpartition(".")[0]).write_text(out)


if __name__ == "__main__":
    generate_bindings()
