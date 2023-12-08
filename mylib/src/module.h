#pragma once
#include <optix_types.h>

#include <torch/extension.h>
#include "context.h"

class Module
{
    OptixDeviceContext context;

    OptixModule module = {};
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};

    OptixProgramGroupOptions program_group_options = {};
    std::array<OptixProgramGroupDesc, 3> program_group_descriptions = {};
    std::array<OptixProgramGroup, 3> program_groups = {nullptr};

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipeline_link_options = {.maxTraceDepth = 1};
    uint32_t max_traversal_depth = 1; // nested bvhs

public:
    Module(Context &context);

    void load(std::string ptx_source);
};

void bind_module(py::module &m);