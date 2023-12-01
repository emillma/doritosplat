#include "optix_types.h"
#include <torch/extension.h>
PYBIND11_MODULE(structs, m)
{
    py::class_<OptixDisplacementMicromapDesc>(m, "OptixDisplacementMicromapDesc")
        //.def(py::init<unsigned int, unsigned short, unsigned short>())
        .def(py::init<>())
        .def_readwrite("byteOffset", &OptixDisplacementMicromapDesc::byteOffset)
        .def_readwrite("subdivisionLevel", &OptixDisplacementMicromapDesc::subdivisionLevel)
        .def_readwrite("format", &OptixDisplacementMicromapDesc::format)
;
}