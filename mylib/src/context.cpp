#include <optix.h>
#include <optix_function_table_definition.h>
#include <sutil/Exception.h>
#include <optix_stubs.h>

#include <iostream>
#include <iomanip>

#include <torch/extension.h>

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

void bind_context(py::module &m)
{
    py::class_<Context>(m, "Context")
        .def(py::init<>());
    // .def("get_context", &Context::get_context);
}