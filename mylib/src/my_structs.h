#pragma once
#include <optix_types.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

typedef struct
{
    CUdeviceptr output_buffer;
    unsigned int width;
    CUdeviceptr input_data;
    OptixTraversableHandle handle;
} Params;

typedef struct
{
} RayGenData;

typedef struct
{
    CUdeviceptr color;
} MissData;

typedef struct
{
} HitData;
