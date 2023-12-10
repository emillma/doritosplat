//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>
#include "my_structs.h"

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ float3 computeRay(uint3 idx, uint3 dim)
{
    const float2 d = -1.f + 2.f * make_float2(
                                      (float)(idx.x) / (float)(dim.x),
                                      (float)(idx.y) / (float)(dim.y));
    const float3 ray_direction = normalize(make_float3(d, 1.0f));
    return ray_direction;
}

extern "C" __global__ void __raygen__rg()
{
    Image out(params.output_buffer);
    // Lookup our location within the launch grid

    //  new (&out) OutputBuffer((float *)params.output_buffer);

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 origin = {0.f, 0.f, 0.f};
    float3 ray_direction = computeRay(idx, dim);

    unsigned int p0, p1, p2 = 0;
    optixTrace(
        params.handle,
        origin,
        ray_direction,
        0.0f,                     // Min intersection distance
        1e16f,                    // Max intersection distance
        0.0f,                     // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0, // SBT offset   -- See SBT discussion
        1, // SBT stride   -- See SBT discussion
        0, // missSBTIndex -- See SBT discussion
        p0,
        p1,
        p2);
    // float3 result;
    // result.x = __uint_as_float(p0);
    // result.y = __uint_as_float(p1);
    // result.z = __uint_as_float(p2);

    // Record results in our output raster

    out(idx.x, idx.y, 0) = p0;
    out(idx.x, idx.y, 3) = p0;
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0);
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1);
}
