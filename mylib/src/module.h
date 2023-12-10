#pragma once
#include <optix_types.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "context.h"
#include "config.h"

class Module
{
    OptixDeviceContext context;

    OptixModule module;
    OptixModuleCompileOptions module_compile_options = {
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
    };

    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipeline_compile_options{
        .usesMotionBlur = 0,
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        .numPayloadValues = 3,
        .numAttributeValues = 3,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
    };
    OptixPipelineLinkOptions pipeline_link_options = {.maxTraceDepth = 1};

    std::vector<OptixProgramGroup> program_groups;
    OptixShaderBindingTable sbt = {};

    unsigned int maxCCDepth = 1;
    unsigned int maxDCDEpth = 1;
    uint32_t max_traversal_depth = 1; // nested bvhs

public:
    Module(Context &context);

    void load_ptx(std::string ptx_source);
    void configure();
    std::vector<std::array<uint8_t, OPTIX_SBT_RECORD_HEADER_SIZE>> get_sbt_headers();
    void generate_sbt(torch::Tensor &raygen_record,
                      torch::Tensor &hitgroup_record,
                      torch::Tensor &miss_record);
    void launch(torch::Tensor &params, size_t stream);
};

void bind_module(py::module &m);