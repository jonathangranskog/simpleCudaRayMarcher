#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"

#include <stdio.h>
#include <iostream>
#include "lodepng.h"
#include "sdf_util.hpp"

#define BLOCK_SIZE 8
#define EPS 1e-5
#define PUSH 1e-2

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

float __host__ __device__ DE(float3 pos) 
{
	float mb = mandelbulb(pos / 2.3f, 8, 4, 8.0f) * 2.3f;
	float sphere1 = sdfSphere(pos - make_float3(1.5f, 0.0f, 1.0f), 1.0f);
	float sphere2 = sdfSphere(pos - make_float3(1.0, 3.0f, 1.0f), 1.0f);
	float spheres = sdfUnion(sphere1, sphere2);
	float plane = sdfPlane(pos - make_float3(0, -1.0f, 0), make_float3(0, 1, 0));
	return sdfUnion(mb, sdfUnion(plane, spheres));
}

__device__ Hit march(float3 orig, float3 direction) 
{
	float totaldist = 0.0f;
	float maxdist = length(direction);
	float3 pos = orig; float3 dir = normalize(direction);

	Hit hit;

	while (totaldist < maxdist) 
	{
		float t = DE(pos);
		if (t < 0.005f) 
		{
			float fx = (DE(make_float3(pos.x + EPS, pos.y, pos.z)) - DE(make_float3(pos.x - EPS, pos.y, pos.z))) / (2.0f * EPS);
			float fy = (DE(make_float3(pos.x, pos.y + EPS, pos.z)) - DE(make_float3(pos.x, pos.y - EPS, pos.z))) / (2.0f * EPS);
			float fz = (DE(make_float3(pos.x, pos.y, pos.z + EPS)) - DE(make_float3(pos.x, pos.y, pos.z - EPS))) / (2.0f * EPS);
			float3 normal = normalize(make_float3(fx - t, fy - t, fz - t));
			if (dot(-dir, normal) < 0) normal = -normal;
			hit.isHit = true;
			hit.pos = pos;
			hit.normal = normal;
			hit.color = make_float3(1.0f, 1.0f, 1.0f);
			return hit;
		}

		totaldist += t;
		pos += t * dir;
	}

	return hit;
}

__device__ float3 trace(float3 orig, float3 direction)
{
	Hit rayhit = march(orig, direction);
	
	float3 color = (rayhit.isHit) ? rayhit.color * dot(rayhit.normal, -normalize(direction)) : make_float3(0, 0, 0);
	return color;
}

__global__ void render(int width, int height, float* result, Camera cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	float2 offset = make_float2(0.5f, 0.5f);
	float2 sample = make_float2(x, y) + offset;
	float nx = (sample.x / float(width) - 0.5f) * 2.0f;
	float ny = -(sample.y / float(height) - 0.5f) * 2.0f;
	ny *= float(height) / float(width);
	float3 pt = cam.pos + cam.side * cam.halffov * nx + cam.up * ny * cam.halffov + cam.dir;
	float3 raydir = normalize(pt - cam.pos);
	
	float3 color = trace(cam.pos, raydir * cam.maxdist);

	result[x * 3 + 3 * y * width + 0] = color.x;
	result[x * 3 + 3 * y * width + 1] = color.y;
	result[x * 3 + 3 * y * width + 2] = color.z;
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
	cam.pos = make_float3(-1, 1.5, -3);
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
