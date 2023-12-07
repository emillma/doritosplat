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
#include <vector>

// #include "c_ptr_types.h"
// #include "myclass.h"

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

class Context
{
    OptixDeviceContext context = nullptr;

public:
    Context()
    {
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {
            .logCallbackFunction = &context_log_cb,
            .logCallbackLevel = 4};
        OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));
    };
    ~Context() { OPTIX_CHECK(optixDeviceContextDestroy(context)); };
    OptixDeviceContext &get_context() { return context; };
};

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
    auto get_triangle_input() { return triangle_input; };

    Scene(Context &context) : context(context.get_context())
    {
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    };

    OptixAccelBufferSizes set_vertex_pointer(torch::Tensor vertices)
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

    void build(torch::Tensor temp_buffer,
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
};

class Module
{
    OptixDeviceContext context;

    OptixModule module = {};
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};

    OptixProgramGroupOptions program_group_options = {};
    std::array<OptixProgramGroupDesc, 3> program_group_descriptions = {};
    std::array<OptixProgramGroup, 3> program_groups = {nullptr};

public:
    Module(Context &context) : context(context.get_context())
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

    void load(std::string ptx_source)
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
    };
};

PYBIND11_MODULE(c_optix, m)
{

    py::class_<Context>(m, "Context")
        .def(py::init<>());

    py::class_<Scene>(m, "Scene")
        .def(py::init<Context &>())
        .def("set_vertex_pointer", &Scene::set_vertex_pointer)
        .def("build", &Scene::build)
        .def_property_readonly("triangle_input", &Scene::get_triangle_input);

    py::class_<Module>(m, "Module")
        .def(py::init<Context &>())
        .def("load", &Module::load);
}