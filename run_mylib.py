import torch
from mylib import structs, enums
from mylib.c_bindings import c_optix


# context = c_optix.create_context()


vertices = torch.tensor(
    [
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.0, 0.5, 0],
    ],
    device="cuda",
)
context = c_optix.create_context()
a, b = c_optix.triangle_setup(context, vertices)
build_config: structs.OptixBuildInput = a
buffer_sizez: structs.OptixAccelBufferSizes = b

accelerated_buffer = torch.zeros(buffer_sizez.outputSizeInBytes, device="cuda")
build_buffer = torch.zeros(buffer_sizez.tempSizeInBytes, device="cuda")

print("hello")
here = True
# stubs.optixAccelComputeMemoryUsage(
# mylib.cmylib.optixAccelComputeMemoryUsage(ctx)
# print(mylib.myfunc("hello"))
