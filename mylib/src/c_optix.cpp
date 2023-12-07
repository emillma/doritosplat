// #include <optix.h>

#include <torch/extension.h>
#include "context.h"
#include "scene.h"
// #include "module.h"

PYBIND11_MODULE(c_optix, m)
{
    bind_context(m);
    bind_scene(m);
    // bind_module(m);
    // py::class_<Context>(m, "Context")
    //     .def(py::init<>());

    // py::class_<Module>(m, "Module")
    //     .def(py::init<Context &>())
    //     .def("load", &Module::load);
}