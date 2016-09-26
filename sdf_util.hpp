#ifndef __SDF_UTIL__
#define __SDF_UTIL__

#include <cuda_runtime.h>
#include "cutil_math.h"

inline float __host__ __device__ sdfSphere(float3 pos, float radius)
{
	return length(pos) - radius;
}

#endif