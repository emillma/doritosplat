#include <optix.h>
class Context
{
public:
    OptixDeviceContext &get_context();
};

void bind_context(py::module &m);
