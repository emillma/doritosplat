import mylib
import torch
from mylib import types
from mylib.bindings import create_context

ctx = create_context()

vertices = torch.tensor(
    [
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.0, 0.5, 0],
    ],
    device="cuda",
)
vertices.data_ptr()

# print(mylib.myfunc("hello"))
