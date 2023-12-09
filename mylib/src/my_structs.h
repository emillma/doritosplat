#pragma once

typedef struct
{
    float *output_buffer;
} Params;

typedef struct
{
} RayGenData;

typedef struct
{
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
