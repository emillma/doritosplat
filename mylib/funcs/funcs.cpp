#include <torch/extension.h>
#include <optix_types.h>

OptixDisplacementMicromapDesc foo(OptixDisplacementMicromapDesc &desc)
{
    desc.format = 2;
    return desc;
}

PYBIND11_MODULE(funcs, m)
{
    m.def("foo", &foo);
}