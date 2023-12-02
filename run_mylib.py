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

accel_options = types.OptixAccelBuildOptions(
    buildFlags=types.OPTIX_BUILD_FLAG_NONE,
    operation=types.OPTIX_BUILD_OPERATION_BUILD,
)

triangle_input_flags = torch.IntTensor([types.OPTIX_GEOMETRY_FLAG_NONE])
triangle_input = types.OptixBuildInput(
    type=types.OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
    triangleArray=types.OptixBuildInputTriangleArray(
        vertexFormat=types.OPTIX_VERTEX_FORMAT_FLOAT3,
        numVertices=vertices.shape[0],
        vertexBuffers=vertices.data_ptr(),
        flags=triangle_input_flags.data_ptr(),
        numSbtRecords=1,
    ),
)
vertices.data_ptr()

# print(mylib.myfunc("hello"))
