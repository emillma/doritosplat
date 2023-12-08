#include <sutil/Exception.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "module.h"

Module::Module(Context &context) : context(context.get_context())
{
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 3;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
};

void Module::load(std::string ptx_source)
{

    OPTIX_CHECK_LOG(optixModuleCreate(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_source.c_str(),
        ptx_source.size(),
        LOG,
        &LOG_SIZE,
        &module));

    program_group_descriptions[0] = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN, .raygen = {.module = module, .entryFunctionName = "__raygen__rg"}};
    program_group_descriptions[1] = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS, .miss = {.module = module, .entryFunctionName = "__miss__ms"}};
    program_group_descriptions[2] = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP, .hitgroup = {.moduleCH = module, .entryFunctionNameCH = "__closesthit__ch"}};

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context,
        program_group_descriptions.data(),
        (uint32_t)program_group_descriptions.size(),
        &program_group_options,
        LOG,
        &LOG_SIZE,
        program_groups.data()));

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        (uint32_t)program_groups.size(),
        LOG, &LOG_SIZE,
        &pipeline));

    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup &prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
    };

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, pipeline_link_options.maxTraceDepth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));
};

void bind_module(py::module &m)
{
    py::class_<Module>(m, "Module")
        .def(py::init<Context &>())
        .def("load", &Module::load);
}