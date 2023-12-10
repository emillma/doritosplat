#pragma once
#include <optix_types.h>
#include "../src/config.h"

#define CU __forceinline__ __device__

template <typename T, int _Rows, int _Cols, int _Channels>
class Matrix
{
private:
    T *ptr;

public:
    CU Matrix(size_t ptr_int) : ptr((T *)ptr_int){};
    CU T &operator()(int row, int col, int channel)
    {
        return ptr[row * _Cols * _Channels + col * _Channels + channel];
    };
};

typedef Matrix<float, IMGX, IMGY, 4> Image;

typedef struct Params
{
    OptixTraversableHandle handle;
    size_t output_buffer;
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
