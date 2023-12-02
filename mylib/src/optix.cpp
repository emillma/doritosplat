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
#include "mytypes.h"
static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

size_t create_context()
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
    return (size_t)context;
}

void change_int(int &a)
{
    a = 100;
}

PYBIND11_MODULE(c_optix, m)
{
    m.def("create_context", &create_context);
    m.def("change_int", &change_int);
    m.def("change_int", [](int i)
          { change_int(i); return std::make_tuple(i); });
}