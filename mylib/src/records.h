#pragma once
#include <optix_types.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

struct Params
{
    uchar4 *image;
    unsigned int image_width;
    unsigned int image_height;
    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

struct RayGenData
{
};

struct MissData
{
    float3 bg_color;
};

struct HitGroupData
{
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

void bind_records(py::module &m);
