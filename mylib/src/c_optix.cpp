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
#include "my_types.h"

template <class T>
class ptr_wrapper
{
public:
    ptr_wrapper() : ptr(nullptr) {}
    ptr_wrapper(T *ptr) : ptr(ptr) {}
    ptr_wrapper(const ptr_wrapper &other) : ptr(other.ptr) {}
    void destroy() { delete ptr; }
    T &operator*() const { return *ptr; }
    T *operator->() const { return ptr; }
    T *get() const { return ptr; }
    T &operator[](std::size_t idx) const { return ptr[idx]; }

private:
    T *ptr;
};

typedef ptr_wrapper<OptixDeviceContext_t> OptixDeviceContext_pyptr;

static void
context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

OptixDeviceContext create_context()
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
    return context;
}
void destroy_context(OptixDeviceContext context)
{
    optixDeviceContextDestroy(context);
}
PYBIND11_MODULE(c_optix, m)
{
    py::class_<OptixDeviceContext_pyptr>(m, "pyOptixDeviceContext")
        .def(py::init<>());

    m.def(
        "create_context", []()
        { return OptixDeviceContext_pyptr(create_context()); },
        py::return_value_policy::take_ownership);
    m.def(
        "destroy_context", [](OptixDeviceContext_pyptr context)
        { return destroy_context(context.get()); },
        py::return_value_policy::take_ownership);
}