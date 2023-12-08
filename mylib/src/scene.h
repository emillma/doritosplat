#pragma once
#include <optix_types.h>
// #include <sutil/Exception.h>

#include <torch/extension.h>
#include "context.h"

class Scene
{
    OptixDeviceContext context;
    CUdeviceptr d_vertices = {};
    OptixAccelBuildOptions accel_options = {};
    OptixAccelBufferSizes gas_buffer_sizes = {};
    OptixTraversableHandle gas_handle = {};
    OptixBuildInput triangle_input = {};

    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

public:
    Scene(Context &context);

    OptixAccelBufferSizes set_vertex_pointer(torch::Tensor vertices);

    void build(torch::Tensor temp_buffer,
               torch::Tensor output_buffer);
};

void bind_scene(py::module &m);
