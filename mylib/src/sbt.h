#pragma once
#include <optix_types.h>
#include "records.h"
#include "context.h"
class Sbt
{
    OptixShaderBindingTable sbt = {};

public:
    Sbt(Context &context);
    OptixShaderBindingTable &get_sbt();
}