#pragma once
#include <optix.h>
#include <optix_stubs.h>
// #include <optix_function_table_definition.h>
// #include <optix_stack_size.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

void setup_context(OptixDeviceContext &context);
