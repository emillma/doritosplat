#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <iostream>
#include <iomanip>
#include <torch/extension.h>
#include "mytypes.h"
static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

MyContext create_context()
{
    OptixDeviceContext context = nullptr;
    {
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());
        CUcontext cuda_context = 0; // zero means take the current context
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &options, &context));
    }
    return MyContext{context = context};
}

// void create_gas(torch::Tensor vertices)
// {
//     OptixTraversableHandle gas_handle;
//     CUdeviceptr d_gas_output_buffer;
//     {
//         // Use default options for simplicity.  In a real use case we would want to
//         // enable compaction, etc
//         OptixAccelBuildOptions accel_options = {};
//         accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
//         accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

//         // Triangle build input: simple list of three vertices
//         // const std::array<float3, 3> vertices =
//         //     {{{-0.5f, -0.5f, 0.0f},
//         //       {0.5f, -0.5f, 0.0f},
//         //       {0.0f, 0.5f, 0.0f}}};

//         // const size_t vertices_size = sizeof(float3) * vertices.size();
//         // CUdeviceptr d_vertices = 0;
//         // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
//         // CUDA_CHECK(cudaMemcpy(
//         //     reinterpret_cast<void *>(d_vertices),
//         //     vertices.data(),
//         //     vertices_size,
//         //     cudaMemcpyHostToDevice));

//         // Our build input is a simple list of non-indexed triangle vertices

//         const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
//         OptixBuildInput triangle_input = {};
//         triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
//         triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
//         triangle_input.triangleArray.numVertices = vertices.size(0);
//         triangle_input.triangleArray.vertexBuffers = (CUdeviceptr *)vertices.data_ptr();
//         triangle_input.triangleArray.flags = triangle_input_flags;
//         triangle_input.triangleArray.numSbtRecords = 1;

//         OptixAccelBufferSizes gas_buffer_sizes;
//         OPTIX_CHECK(optixAccelComputeMemoryUsage(
//             context,
//             &accel_options,
//             &triangle_input,
//             1, // Number of build inputs
//             &gas_buffer_sizes));
//         CUdeviceptr d_temp_buffer_gas;
//         CUDA_CHECK(cudaMalloc(
//             reinterpret_cast<void **>(&d_temp_buffer_gas),
//             gas_buffer_sizes.tempSizeInBytes));
//         CUDA_CHECK(cudaMalloc(
//             reinterpret_cast<void **>(&d_gas_output_buffer),
//             gas_buffer_sizes.outputSizeInBytes));

//         OPTIX_CHECK(optixAccelBuild(
//             context,
//             0, // CUDA stream
//             &accel_options,
//             &triangle_input,
//             1, // num build inputs
//             d_temp_buffer_gas,
//             gas_buffer_sizes.tempSizeInBytes,
//             d_gas_output_buffer,
//             gas_buffer_sizes.outputSizeInBytes,
//             &gas_handle,
//             nullptr, // emitted property list
//             0        // num emitted properties
//             ));

//         // We can now free the scratch space buffer used during build and the vertex
//         // inputs, since they are not needed by our trivial shading method
//         CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
//         CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
//     }
// }
PYBIND11_MODULE(cmylib, m)
{
    m.def("create_context", &create_context);
}