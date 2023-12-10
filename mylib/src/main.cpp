// #include <optix.h>

// #include <torch/extension.h>
#include <optix_function_table_definition.h>

#include "context.h"
// #include "optix_function_table.h"
#include "scene.h"
#include "module.h"
// #include "../generated/enums.h"
// #include "../generated/structs.h"

#include "../generated/generated_bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      bind_structs(m);
      bind_enums(m);

      bind_context(m);
      bind_scene(m);
      bind_module(m);

      m.def("sizeof_OptixTraversableHandle", []()
            { return sizeof(OptixTraversableHandle); });

      m.def(
          "test_vector", []()
          {
        std::vector<int> v;
        v.push_back(1);
        v.push_back(2);
        return v; },
          py::return_value_policy::take_ownership);

      m.def("modify_vector", [](std::vector<int> &v)
            { v[0] = 3; return v; });

      m.def("at_row_1", [](torch::Tensor &t)
            { return (size_t)t[1].data_ptr(); });

      // py::class_<Context>(m, "Context")
      //     .def(py::init<>());

      // py::class_<Module>(m, "Module")
      //     .def(py::init<Context &>())
      //     .def("load", &Module::load);
}