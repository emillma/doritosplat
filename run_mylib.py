import torch
from mylib import structs, enums
from mylib.lib_loader import get_optixir
from mylib.c_bindings import c_optix


# context = c_optix.create_context()
ir = get_optixir("/workspaces/doritosplat/insp/optixTriangle/optixTriangle.cu")

vertices = torch.tensor(
    [
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.0, 0.5, 0],
    ],
    device="cuda",
)
assert vertices.data_ptr() % 64 == 0
context = c_optix.Context()

scene = c_optix.Scene(context)
sizes: structs.OptixAccelBufferSizes = scene.set_vertex_pointer(vertices)
tmp = torch.empty(sizes.tempSizeInBytes, dtype=torch.uint8, device="cuda")
bvh = torch.empty(sizes.outputSizeInBytes, dtype=torch.uint8, device="cuda")
scene.build(tmp, bvh)

module = c_optix.Module(context)
module.load(ir)


header = [0] * 64

a = torch.empty(24, dtype=torch.uint8, device="cpu")
sizes.copy_to_tensor(a)
sezes2 = type(sizes).from_tensor(a)
torch.tensor(
    [*([0] * 64)],
    dtype=torch.int64,
    device="cuda",
)
