#ifndef __SDF_UTIL__
#define __SDF_UTIL__

#include <cuda_runtime.h>
#include "cutil_math.h"

inline float __host__ __device__ sdfUnion(float a, float b)
{
	return min(a, b);
}

inline float __host__ __device__sdfDifference(float a, float b)
{
	return max(-a, b);
}

inline float __host__ __device__ sdfIntersection(float a, float b)
{
	return max(a, b);
}

inline float __host__ __device__ sdfSphere(float3 pos, float radius)
{
	return length(pos) - radius;
}

inline float __host__ __device__ sdfPlane(float3 pos, float3 n)
{
	return dot(pos, n);
}

inline float __host__ __device__ mandelbulb(float3 pos, int iterations, float bail, float power)
{
	float3 z = pos;
	float dr = 1.0;
	float r = 0.0;
	for (int i = 0; i < iterations; i++) {
		r = length(z);
		if (r > bail) break;

		// convert to polar coordinates
		float theta = asin(z.z / r);
		float phi = atan2(z.y, z.x);
		dr = pow(r, power - 1.0f) * power * dr + 1.0f;

		// scale and rotate the point
		float zr = pow(r, power);
		theta = theta * power;
		phi = phi * power;

		// convert back to cartesian coordinates
		z = zr * make_float3(cos(theta)*cos(phi), sin(phi)*cos(theta), sin(theta));
		z += pos;
	}

	return 0.5*log(r)*r / dr;
}

#endif