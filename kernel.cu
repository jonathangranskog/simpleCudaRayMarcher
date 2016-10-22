#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"

#include <stdio.h>
#include <iostream>
#include "lodepng.h"
#include "sdf_util.hpp"

#define BLOCK_SIZE 8
#define BOUNCES 2
#define SAMPLES 64
#define EPS 1e-5
#define MINDIST 0.5e-2
#define PUSH MINDIST*2

// TODO: Improve sampling and random function, figure out how to use CURAND?
inline float __host__ __device__ myrand(const float2& seed) 
{
	// Really simple "random"
	float s = abs(sin(dot(seed, make_float2(12.9898f, 78.233f))) * 43758.5453f);
	float frac = s - int(s);
	return frac;
}

// Purely random pixel sample
inline float2 __host__ __device__ getRandomSample(int i, const float2& seed) 
{
	float2 s = make_float2(i + (i + 2) * seed.y + (i + 1) * seed.x, seed.x + seed.y + (i + 1));
	return make_float2(myrand(s), myrand(s + 52));
}

float3 __host__ __device__ orient(const float3& n, const float2& seed) 
{
	// rejection sampling hemisphere
	float x = 1.0f, y = 1.0f;
	float i = 0.0f;
	float j = 0.0f;
	while (x * x + y * y > 1.0f) 
	{
		x = (myrand(seed + make_float2(i * 1.9f, i * j * 3.72f + 0.5f)) - 0.5f) * 2.0f;
		y = (myrand(seed + make_float2(4.5f * j, 3.5f + 0.7333f * j * j)) - 0.5f) * 2.0f;
		i += seed.x;
		j += seed.y;
	}
	float z = sqrtf(1 - x * x - y * y);
	float3 in = normalize(make_float3(x, y, z));

	// Create vector that is not the same as n
	float3 absn = fabs(n);
	float3 q = n;
	if (absn.x <= absn.y && absn.x <= absn.z)  q.x = 1;
	else if (absn.y <= absn.x && absn.y <= absn.z) q.y = 1;
	else q.z = 1;

	// Basis creation, result is just a rolled out matrix multiplication of basis matrix and in vector
	float3 t = normalize(cross(n, q));
	float3 b = normalize(cross(n, t));
	return normalize(make_float3(t.x * in.x + b.x * in.y + n.x * in.z,
								 t.y * in.x + b.y * in.y + n.y * in.z,
								 t.z * in.x + b.z * in.y + n.z * in.z));
}

struct Hit 
{
	bool isHit = 0;
	float3 pos;
	float3 normal;
	float3 color;
};

struct Camera 
{
	float3 pos;
	float3 dir;
	float halffov;
	float maxdist = 20.0f;
	float3 up;
	float3 side;
};


// Distance estimation function
float __host__ __device__ DE(const float3& pos) 
{
	//return mandelbulbScene(pos);
	return sphereScene(pos);
}

// Ray marching function, similar to intersect function in normal ray tracers
__host__ __device__ Hit march(const float3& orig, const float3& direction) 
{
	float totaldist = 0.0f;
	float maxdist = length(direction);
	float3 pos = orig; float3 dir = normalize(direction);

	Hit hit;

	while (totaldist < maxdist) 
	{
		float t = DE(pos);

		// If distance is less than this then it is a hit.
		if (t < MINDIST) 
		{
			// Calculate gradient (normal)
			float fx = (DE(make_float3(pos.x + EPS, pos.y, pos.z)) - DE(make_float3(pos.x - EPS, pos.y, pos.z))) / (2.0f * EPS);
			float fy = (DE(make_float3(pos.x, pos.y + EPS, pos.z)) - DE(make_float3(pos.x, pos.y - EPS, pos.z))) / (2.0f * EPS);
			float fz = (DE(make_float3(pos.x, pos.y, pos.z + EPS)) - DE(make_float3(pos.x, pos.y, pos.z - EPS))) / (2.0f * EPS);
			float3 normal = normalize(make_float3(fx - t, fy - t, fz - t));
			// faceforward
			if (dot(-dir, normal) < 0) normal = -normal;

			// create hit
			hit.isHit = true;
			hit.pos = pos;
			hit.normal = normal;
			hit.color = make_float3(0.85f, 0.85f, 0.85f);
			return hit;
		}

		// step forwards by t if no hit
		totaldist += t;
		pos += t * dir;
	}

	return hit;
}

// Path tracing function
__host__ __device__ float3 trace(const float3& orig, const float3& direction, const float2& seed)
{
	float raylen = length(direction);
	float3 dir = direction;
	float3 o = orig;
	float3 p = make_float3(0.0f); float3 n = make_float3(0.0f);
	float3 mask = make_float3(1.0f); float3 color = make_float3(0.0f);

	Hit rayhit = march(o, dir);
	
	for (int i = 0; i < BOUNCES + 1; i++) 
	{
		if (rayhit.isHit) 
		{
			p = rayhit.pos; n = rayhit.normal;
			// Create new ray direction
			float3 d = orient(n, (i + 1) * seed * 13.735791f);
			o = p + n * PUSH;
			mask *= rayhit.color;
			dir = raylen * d;
			// Fire new ray if there are bounces left
			if (i < BOUNCES) rayhit = march(o, dir);
		}
		else if (i == 0) return make_float3(0.0f); // black background
		else 
		{
			color += make_float3(1.0f) * mask; // add color when light (sky) is hit
			break;
		}
	}
	
	return color;
}

__global__ void render(int width, int height, float* result, Camera cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	// Store colors in shared memory for faster read/write time
	__shared__ float3 colors[BLOCK_SIZE * BLOCK_SIZE];
	int idx = threadIdx.x * blockDim.y + threadIdx.y;
	colors[idx] = make_float3(0.0f);

	float2 samp = make_float2(x, y);
	float2 seed = make_float2(samp.x * 1.733f + samp.y * samp.x * 3.5150f, samp.y * 1.572f + 2.8349f * samp.x * samp.x);

	for (int i = 0; i < SAMPLES; i++) {
		float2 offset = getRandomSample(i, seed);
		float2 sample = samp + offset;
		float nx = (sample.x / float(width) - 0.5f) * 2.0f;
		float ny = -(sample.y / float(height) - 0.5f) * 2.0f;
		ny *= float(height) / float(width);
		float3 pt = cam.pos + cam.side * cam.halffov * nx + cam.up * ny * cam.halffov + cam.dir;
		float3 raydir = normalize(pt - cam.pos);
		colors[idx] += trace(cam.pos, raydir * cam.maxdist, seed + offset);
	}
	
	colors[idx] /= SAMPLES;

	result[x * 3 + 3 * y * width + 0] = colors[idx].x;
	result[x * 3 + 3 * y * width + 1] = colors[idx].y;
	result[x * 3 + 3 * y * width + 2] = colors[idx].z;
}

void saveImage(int width, int height, const float colors[]) 
{
	std::vector<unsigned char> output;
	output.resize(4 * width * height);
	for (int i = 0; i < width * height; i++)
	{
		output[i * 4 + 0] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 3 + 0] * 255, 255), 0));
		output[i * 4 + 1] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 3 + 1] * 255, 255), 0));
		output[i * 4 + 2] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 3 + 2] * 255, 255), 0));
		output[i * 4 + 3] = 255;
	}
	unsigned error = lodepng::encode("test.png", output, width, height);
	if (error) std::cout << "An error occurred: " << lodepng_error_text(error) << std::endl;
}

int main()
{
	int width = 1024, height = 1024;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	float *deviceImage;
	cudaMalloc(&deviceImage, 3 * width * height * sizeof(float));

	Camera cam;
	cam.pos = make_float3(-1, 1.5f, -3);
	cam.dir = normalize(-cam.pos);
	cam.side = normalize(cross(cam.dir, make_float3(0, 1, 0)));
	cam.up = normalize(cross(cam.side, cam.dir));
	float fov = 90.0f;
	cam.halffov = std::tan(fov / 2.0f);

	render << <blocks, threads >> >(width, height, deviceImage, cam);

	float *hostImage = (float*) malloc(3 * width * height * sizeof(float));
	cudaMemcpy(hostImage, deviceImage, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost);
	saveImage(width, height, hostImage);
	cudaFree(deviceImage);
	free(hostImage);
    return 0;
}
