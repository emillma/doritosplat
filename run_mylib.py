import mylib
import torch
from mylib import structs, enums, c_bindings, stubs
import ctypes

# ctx = create_context()

vertices = torch.tensor(
    [
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.0, 0.5, 0],
    ],
    device="cuda",
)

accel_options = structs.OptixAccelBuildOptions(
    buildFlags=enums.OPTIX_BUILD_FLAG_NONE,
    operation=enums.OPTIX_BUILD_OPERATION_BUILD,
)

triangle_input_flags = torch.IntTensor([enums.OPTIX_GEOMETRY_FLAG_NONE])
triangle_input = structs.OptixBuildInput(
    type=enums.OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
    triangleArray=structs.OptixBuildInputTriangleArray(
        vertexFormat=enums.OPTIX_VERTEX_FORMAT_FLOAT3,
        numVertices=vertices.shape[0],
        vertexBuffers=vertices.data_ptr(),
        flags=triangle_input_flags.data_ptr(),
        numSbtRecords=1,
    ),
)
a = 10
stubs.optixProgramGroupCreate()
# mylib.cmylib.optixAccelComputeMemoryUsage(ctx)
# print(mylib.myfunc("hello"))
