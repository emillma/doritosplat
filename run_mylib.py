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

context = c_optix.Context()

scene = c_optix.Scene(context)
sizes: structs.OptixAccelBufferSizes = scene.set_vertex_pointer(vertices)
tmp = torch.empty(sizes.tempSizeInBytes, dtype=torch.uint8, device="cuda")
bvh = torch.empty(sizes.outputSizeInBytes, dtype=torch.uint8, device="cuda")
scene.build(tmp, bvh)

module = c_optix.Module(context)
module.load(ir)

# here = True
# a, b = c_optix.triangle_setup(context, vertices)
# build_config: structs.OptixBuildInput = a
# buffer_sizez: structs.OptixAccelBufferSizes = b

# accelerated_buffer = torch.zeros(buffer_sizez.outputSizeInBytes, device="cuda")
# build_buffer = torch.zeros(buffer_sizez.tempSizeInBytes, device="cuda")

# print("hello")
# here = True
# stubs.optixAccelComputeMemoryUsage(
# mylib.cmylib.optixAccelComputeMemoryUsage(ctx)
# print(mylib.myfunc("hello"))
