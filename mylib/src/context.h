#pragma once

#include <sutil/Exception.h>

#include <torch/extension.h>

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */);

void bind_context(py::module &m);
class Context
{
    OptixDeviceContext context = nullptr;

public:
    Context();
    ~Context();
    OptixDeviceContext &get_context();
};
