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
} HitGroupData;

// typedef struct
// {
//     HitGroupData data;
// } HitGroupSbtRecord;

// typedef SbtRecord<RayGenData> RayGenSbtRecord;

// typedef SbtRecord<MissData> MissSbtRecord;

// typedef SbtRecord<HitGroupData> HitGroupSbtRecord;
// template <typename T>
// struct SbtRecord
// {
// __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
//     T data;
// };
