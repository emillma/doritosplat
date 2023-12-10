#include <optix_stubs.h>
#include <sutil/Exception.h>
#include "scene.h"

Scene::Scene(Context &context) : context(context.get_context())
{
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
};

OptixAccelBufferSizes Scene::set_vertex_pointer(torch::Tensor vertices)
{
    d_vertices = (CUdeviceptr)vertices.data_ptr();
    triangle_input.triangleArray.numVertices = (uint32_t)vertices.size(1);
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &triangle_input,
        1, // Number of build inputs
        &gas_buffer_sizes));
    return gas_buffer_sizes;
};

void Scene::build(torch::Tensor temp_buffer,
                  torch::Tensor output_buffer)
{
    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // CUDA stream
        &accel_options,
        &triangle_input,
        1, // num build inputs
        (CUdeviceptr)temp_buffer.data_ptr(),
        gas_buffer_sizes.tempSizeInBytes,
        (CUdeviceptr)output_buffer.data_ptr(),
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr, // emitted property list
        0        // num emitted properties
        ));
};

void bind_scene(py::module &m)
{
    py::class_<Scene>(m, "Scene")
        .def(py::init<Context &>())
        .def("set_vertex_pointer", &Scene::set_vertex_pointer)
        .def("build", &Scene::build)
        .def_property_readonly("gas_handle", [](Scene &scene)
                               { return scene.gas_handle; });
}