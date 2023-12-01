from pathlib import Path
from dataclasses import dataclass, field, fields
from mylib._loader import load
cpp = load('structs', Path(__file__).parent, False)

@dataclass
class OptixDisplacementMicromapDesc: 
    byteOffset: 'unsigned int' = field(default=0)
    subdivisionLevel: 'unsigned short' = field(default=0)
    format: 'unsigned short' = field(default=0)

    def __new__(cls, **kwargs):
        obj = cpp.OptixDisplacementMicromapDesc()
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj
