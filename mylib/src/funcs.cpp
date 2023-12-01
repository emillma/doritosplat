#include "optix_types.h"
#include "mytypes.h"

#include <torch/extension.h>

int add(int i, int j)
{
    return i + j;
}
PYBIND11_MODULE(cmylib, m)
{
    py::module types = py::module::import("types");
    m.def("add", &add, "A function which adds two numbers");
}