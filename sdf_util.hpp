#ifndef __SDF_UTIL__
#define __SDF_UTIL__

#include <cuda_runtime.h>
#include "cutil_math.h"

inline float __host__ __device__ sdfUnion(float a, float b)
{
	return min(a, b);
}

inline float __host__ __device__ sdfDifference(float a, float b)
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

inline float __host__ __device__ sdfBox(float3 pos, float3 dim) 
{
	float3 d = fabs(pos) - dim;
	return min(max(d.x, max(d.y, d.z)), 0.0f) + length(fmaxf(d, make_float3(0.0f)));
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

	return 0.5f*log(r)*r / dr;
}

inline float3 __host__ __device__ mandelbulbColor(float3 pos, int iterations, float bail, float power)
{
	float3 color = make_float3(1, 0, 1);

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

		color = make_float3(cos(theta * 1.9f)*cos(phi * 1.45f), sin(phi * 1.45f)*cos(theta * 1.2f), sin(theta * 1.2f));

		// scale and rotate the point
		float zr = pow(r, power);
		theta = theta * power;
		phi = phi * power;

		// convert back to cartesian coordinates
		z = zr * make_float3(cos(theta)*cos(phi), sin(phi)*cos(theta), sin(theta));
		z += pos;
	}

	return color;
}

inline float __host__ __device__ mandelbulbScene(const float3& pos) 
{
	float mb = mandelbulb(pos / 2.3f, 8, 4, 8.0f) * 2.3f;
	return mb;
}

inline float3 __host__ __device__ mandelbulbColor(const float3& pos)
{
	return mandelbulbColor(pos / 2.3f, 8, 4, 8.0f);
}

inline float __host__ __device__ sphereScene(const float3& pos) 
{
	float3 mod1 = make_float3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0f) - 1.f);
	float spheres1 = sdfSphere(mod1, 0.5f);

	float3 mod2 = make_float3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0) - 1.f);
	float spheres2 = sdfSphere(mod2, 0.5f);

	float3 mod3 = make_float3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres3 = sdfSphere(mod3, 0.5f);

	float3 mod4 = make_float3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres4 = sdfSphere(mod4, 0.5f);

	float spheres = sdfUnion(sdfUnion(sdfUnion(spheres1, spheres2), spheres3), spheres4);
	float plane = sdfPlane(pos - make_float3(0, -2.0f, 0), make_float3(0, 1, 0));
	return sdfUnion(spheres, plane);
}

inline float3 __host__ __device__ sphereColor(const float3& pos) 
{
	float3 mod1 = make_float3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0f) - 1.f);
	float spheres1 = sdfSphere(mod1, 0.5f);

	float3 mod2 = make_float3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0) - 1.f);
	float spheres2 = sdfSphere(mod2, 0.5f);

	float3 mod3 = make_float3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres3 = sdfSphere(mod3, 0.5f);

	float3 mod4 = make_float3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres4 = sdfSphere(mod4, 0.5f);

	float spheres = sdfUnion(sdfUnion(sdfUnion(spheres1, spheres2), spheres3), spheres4);
	float plane = sdfPlane(pos - make_float3(0, -2.0f, 0), make_float3(0, 1, 0));
	
	if (plane < spheres) return make_float3(1.0f, 0.3f, 0.1f);
	return make_float3(0.85f);
}

inline float __host__ __device__ cornellBoxScene(const float3& pos)
{
	float rightplane = sdfBox(pos - make_float3(-2.0f, 0.0, 0.0), make_float3(0.05f, 2.f, 1.0f));
	float leftplane = sdfBox(pos - make_float3(2.0f, 0.0, 0.0), make_float3(0.05f, 2.f, 1.0f));
	float backplane = sdfBox(pos - make_float3(0.0f, 0.0, 1.0), make_float3(3.0f, 2.f, 0.05f));
	float topplane = sdfBox(pos - make_float3(0.0f, 2.0f, 0.0f), make_float3(2.05f, 0.05f, 1.0f));
	float plane = sdfPlane(pos - make_float3(0, -1.5f, 0), make_float3(0, 1, 0));

	float smallSphere = sdfSphere(pos - make_float3(0.7f, -1.0f, 0.6f), 0.5f);
	float bigSphere = sdfSphere(pos - make_float3(-0.7f, -0.9f, 0.2f), 0.6f);
	float spheres = sdfUnion(bigSphere, smallSphere);

	return sdfUnion(sdfUnion(sdfUnion(sdfUnion(sdfUnion(rightplane, plane), leftplane), backplane), topplane), spheres);
}

inline float3 __host__ __device__ cornellBoxColor(const float3& pos) {
	float rightplane = sdfBox(pos - make_float3(-2.0f, 0.0, 0.0), make_float3(0.05f, 2.f, 1.0f));
	float leftplane = sdfBox(pos - make_float3(2.0f, 0.0, 0.0), make_float3(0.05f, 2.f, 1.0f));
	float backplane = sdfBox(pos - make_float3(0.0f, 0.0, 1.0), make_float3(3.0f, 2.f, 0.05f));
	float topplane = sdfBox(pos - make_float3(0.0f, 2.0f, 0.0f), make_float3(2.05f, 0.05f, 1.0f));
	float plane = sdfPlane(pos - make_float3(0, -1.5f, 0), make_float3(0.0f, 1.0f, 0.0f));

	float smallSphere = sdfSphere(pos - make_float3(0.7f, -1.0f, 0.6f), 0.5f);
	float bigSphere = sdfSphere(pos - make_float3(-0.7f, -0.9f, 0.2f), 0.6f);
	float spheres = sdfUnion(bigSphere, smallSphere);

	float whitewalls = sdfUnion(sdfUnion(topplane, plane), spheres);

	if (leftplane < rightplane && leftplane < whitewalls && leftplane < backplane) {
		return make_float3(0.05f, .5f, 0.8f);
	}
	else if (backplane < rightplane && backplane < whitewalls) {
		return make_float3(1.0f, 0.8f, 0.1f);
	}
	else if (rightplane < whitewalls) {
		return make_float3(.9f, 0.2f, 0.4f);
	}

	return make_float3(0.85f);
}
#endif