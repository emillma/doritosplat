#include <sutil/Exception.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "module.h"

Module::Module(Context &context) : context(context.get_context()){};

void Module::load_ptx(std::string ptx_source)
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
};
void Module::configure()
{

    std::vector<OptixProgramGroupDesc> program_descs = {};
    program_groups = {};

    program_descs.push_back({.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                             .raygen = {.module = module, .entryFunctionName = "__raygen__rg"}});

    program_descs.push_back({.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                             .hitgroup = {.moduleCH = module, .entryFunctionNameCH = "__closesthit__ch"}});

    program_descs.push_back({.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                             .miss = {.module = module, .entryFunctionName = "__miss__ms"}});

    for (OptixProgramGroupDesc &desc : program_descs)
        program_groups.push_back(nullptr);

    OptixProgramGroupOptions program_group_options = {};
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        context,
        program_descs.data(),
        (uint32_t)program_descs.size(),
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
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                           pipeline_link_options.maxTraceDepth,
                                           maxCCDepth, maxDCDEpth,
                                           &direct_callable_stack_size_from_traversal, &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                                          direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size,
                                          max_traversal_depth));
};

std::vector<std::array<uint8_t, OPTIX_SBT_RECORD_HEADER_SIZE>> Module::get_sbt_headers()
{
    std::vector<std::array<uint8_t, OPTIX_SBT_RECORD_HEADER_SIZE>> sbt_headers = {};
    for (auto &program_group : program_groups)
    {
        alignas(16) std::array<uint8_t, OPTIX_SBT_RECORD_HEADER_SIZE> header;
        OPTIX_CHECK(optixSbtRecordPackHeader(program_group, header.data()));
        sbt_headers.push_back(header);
    };
    return sbt_headers;
};

void Module::generate_sbt(torch::Tensor &raygen_record, torch::Tensor &hitgroup_record, torch::Tensor &miss_record)
{
    for (auto &rec : {raygen_record, hitgroup_record, miss_record})
    {
        if (rec.device().type() != torch::kCUDA)
            throw std::runtime_error("records must be on CUDA device");
        if (rec.dtype() != torch::kUInt8)
            throw std::runtime_error("records must be uint8");
    }

    for (auto &rec : {hitgroup_record, miss_record})
    {
        if (rec.dim() != 2)
            throw std::runtime_error("hit and miss records must be 2-dimensional");
        if (rec.stride(0) * rec.dtype().itemsize() % OPTIX_SBT_RECORD_ALIGNMENT != 0)
            throw std::runtime_error("hit and miss records must be OPTIX_SBT_RECORD_ALIGNMENT bytes apart");
    }

    sbt.raygenRecord = (CUdeviceptr)raygen_record.data_ptr();

    sbt.missRecordBase = (CUdeviceptr)miss_record.data_ptr();
    sbt.missRecordStrideInBytes = miss_record.stride(0) * miss_record.dtype().itemsize();
    sbt.missRecordCount = miss_record.size(0);

    sbt.hitgroupRecordBase = (CUdeviceptr)hitgroup_record.data_ptr();
    sbt.hitgroupRecordStrideInBytes = hitgroup_record.stride(0) * hitgroup_record.dtype().itemsize();
    sbt.hitgroupRecordCount = hitgroup_record.size(0);
};

void Module::launch(torch::Tensor &params, size_t stream)
{

    OPTIX_CHECK(optixLaunch(pipeline,
                            (CUstream_st *)stream,
                            (CUdeviceptr)params.data_ptr(),
                            params.nbytes(),
                            &sbt,
                            256,
                            256,
                            1));
};

void bind_module(py::module &m)
{
    py::class_<Module>(m, "Module")
        .def(py::init<Context &>())
        .def("load", &Module::load_ptx)
        .def("configure", &Module::configure)
        .def("get_sbt_headers", &Module::get_sbt_headers)
        .def("generate_sbt", &Module::generate_sbt)
        .def("launch", &Module::launch);
}