#include <iomanip>
#include <iostream>

#include <optix_stubs.h>
#include <sutil/Exception.h>
// #include "optix_function_table.h"

#include "context.h"

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

void bind_context(py::module &m)
{
    py::class_<Context>(m, "Context")
        .def(py::init<>());
}

Context::Context()
{
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {
        .logCallbackFunction = &context_log_cb,
        .logCallbackLevel = 4};
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));
};

Context::~Context()
{
    std::cout << "Context destructor called" << std::endl;
    OPTIX_CHECK(optixDeviceContextDestroy(context));
};

OptixDeviceContext &Context::get_context() { return context; };
