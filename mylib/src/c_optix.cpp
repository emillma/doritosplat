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
#include "c_ptr_types.h"

static void
context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

auto create_context()
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
    return ptr_wrapper<OptixDeviceContext_t>(context);
};

auto triangle_setup(ptr_wrapper<OptixDeviceContext_t> context_ptr, torch::Tensor vertices)

{
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    CUdeviceptr d_vertices = (CUdeviceptr)vertices.data_ptr();
    OptixBuildInput build_config = {};
    build_config.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_config.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_config.triangleArray.numVertices = static_cast<uint32_t>(vertices.size(1));
    build_config.triangleArray.vertexBuffers = &d_vertices;
    build_config.triangleArray.flags = triangle_input_flags;
    build_config.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_ptr.get(),
        &accel_options,
        &build_config,
        1, // Number of build inputs
        &gas_buffer_sizes));
    return py::make_tuple(build_config, gas_buffer_sizes);
};

auto accelerate(ptr_wrapper<OptixDeviceContext_t> context_ptr,
                OptixAccelBuildOptions accel_options,
                OptixBuildInput build_config,
                OptixAccelBufferSizes gas_buffer_sizes,
                CUdeviceptr d_temp_buffer,
                CUdeviceptr d_gas_output_buffer)
{
    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // CUDA stream
        &accel_options,
        &triangle_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr, // emitted property list
        0        // num emitted properties
        ));
};
PYBIND11_MODULE(c_optix, m)
{

    py::class_<ptr_wrapper<OptixDeviceContext_t>>(m, "context_ptr")
        .def(py::init<>(), py::return_value_policy::take_ownership);
    m.def("create_context", &create_context);
    m.def("triangle_setup", &triangle_setup);

    // m.def("destroy_context", &destroy_context);
    // m.def("get_triangle_input", &get_triangle_input);
    // m.def("get_buffer_sizes", &get_buffer_sizes);
}