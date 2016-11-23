#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
#include "lodepng.h"
#include "sdf_util.hpp"

#define BLOCK_SIZE 8
#define BOUNCES 2
#define SAMPLES 2 // Total number of samples is SAMPLES*SAMPLES
#define EPS 1e-5
#define MINDIST 1.8e-3
#define PUSH MINDIST*2
#define M_PI 3.14159265359
#define FRAMES 100

// Purely random pixel sample
inline float2 __device__ getRandomSample(curandState* state) 
{
	return make_float2(curand_uniform(state), curand_uniform(state));
}

// Random sample in nth subpixel
inline float2 __device__ getJitteredSample(int n, curandState* state) {
	float2 rand_vec = make_float2(curand_uniform(state) * (1.0f / SAMPLES), curand_uniform(state) * (1.0f / SAMPLES));
	float2 result = make_float2((n % SAMPLES) * 1.0f / SAMPLES, (n / SAMPLES) * 1.0f / SAMPLES);
	return result + rand_vec;
}

float3 __device__ orient(const float3& n, curandState* state) 
{
	// rejection sampling hemisphere
	float x = 1.0f, y = 1.0f;

	while (x * x + y * y > 1.0f) 
	{
		x = (curand_uniform(state) - 0.5f) * 2.0f;
		y = (curand_uniform(state) - 0.5f) * 2.0f;
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
	float maxdist = 10.0f;
	float3 up;
	float3 side;
};


// Distance estimation function
float __device__ DE(const float3& pos, float time) 
{
	return mandelbulbScene(pos, time);
	//return sphereScene(pos);
	//return cornellBoxScene(pos);
	//return mengerScene(pos, 6);
	//return testFractalScene(pos, time);
}

float3 __device__ sceneColor(const float3& pos, float time) 
{
	return make_float3(0.85f);
	//return mandelbulbColor(pos, time);
	//return sphereColor(pos);
	//return cornellBoxColor(pos);
}

// Ray marching function, similar to intersect function in normal ray tracers
__device__ Hit march(const float3& orig, const float3& direction, float time) 
{
	float totaldist = 0.0f;
	float maxdist = length(direction);
	float3 pos = orig; float3 dir = normalize(direction);
	float3 col = make_float3(0.85f, 0.85f, 0.85f);

	Hit hit;

	while (totaldist < maxdist) 
	{
		float t = DE(pos, time);

		// If distance is less than this then it is a hit.
		if (t < MINDIST) 
		{
			// Calculate gradient (normal)
			float fx = (DE(make_float3(pos.x + EPS, pos.y, pos.z), time) - DE(make_float3(pos.x - EPS, pos.y, pos.z), time)) / (2.0f * EPS);
			float fy = (DE(make_float3(pos.x, pos.y + EPS, pos.z), time) - DE(make_float3(pos.x, pos.y - EPS, pos.z), time)) / (2.0f * EPS);
			float fz = (DE(make_float3(pos.x, pos.y, pos.z + EPS), time) - DE(make_float3(pos.x, pos.y, pos.z - EPS), time)) / (2.0f * EPS);
			float3 normal = normalize(make_float3(fx - t, fy - t, fz - t));
			// faceforward
			if (dot(-dir, normal) < 0) normal = -normal;

			// create hit
			hit.isHit = true;
			hit.pos = pos;
			hit.normal = normal;
			hit.color = sceneColor(pos, time);
			return hit;
		}

		// step forwards by t if no hit
		totaldist += t;
		pos += t * dir;
	}

	return hit;
}

// Path tracing function
__device__ float3 trace(const float3& orig, const float3& direction, curandState* state, float time)
{
	float raylen = length(direction);
	float3 dir = direction;
	float3 o = orig;
	float3 p = make_float3(0.0f); float3 n = make_float3(0.0f);
	float3 mask = make_float3(1.0f); float3 color = make_float3(0.0f);

	Hit rayhit = march(o, dir, time);
	
	for (int i = 0; i < BOUNCES + 1; i++) 
	{
		if (rayhit.isHit) 
		{
			p = rayhit.pos; n = rayhit.normal;
			// Create new ray direction
			float3 d = orient(n, state);
			o = p + n * PUSH;
			mask *= rayhit.color;
			dir = raylen * d;
			// Fire new ray if there are bounces left
			if (i < BOUNCES) rayhit = march(o, dir, time);
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

__global__ void render(int width, int height, float* result, Camera cam, unsigned long long seed, float time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	float3 color = make_float3(0.0f);

	int block = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long long idx = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	curandState state;
	curand_init(idx + seed, 0, 0, &state);

	float2 samp = make_float2(x, y);
	
	for (int i = 0; i < SAMPLES * SAMPLES; i++) {
		//float2 offset = getRandomSample(&state);
		float2 offset = getJitteredSample(i, &state);
		float2 sample = samp + offset;
		float nx = (sample.x / float(width) - 0.5f) * 2.0f;
		float ny = -(sample.y / float(height) - 0.5f) * 2.0f;
		ny *= float(height) / float(width);
		float3 pt = cam.pos + cam.side * cam.halffov * nx + cam.up * ny * cam.halffov + cam.dir;
		float3 raydir = normalize(pt - cam.pos);
		color += trace(cam.pos, raydir * cam.maxdist, &state, time);
	}
	
	color /= (SAMPLES * SAMPLES);

	result[x * 3 + 3 * y * width + 0] = color.x;
	result[x * 3 + 3 * y * width + 1] = color.y;
	result[x * 3 + 3 * y * width + 2] = color.z;
}

void saveImage(std::string path, int width, int height, const float colors[]) 
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
	unsigned error = lodepng::encode(path, output, width, height);
	if (error) std::cout << "An error occurred: " << lodepng_error_text(error) << std::endl;
}

int main()
{
	int width = 1920, height = 1080;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	
	Camera cam;
	cam.pos = make_float3(-1.0f, 1.5f, -3.0f);
	//cam.pos = make_float3(0, 0.4f, -1.4f);
	cam.dir = normalize(-cam.pos);
	cam.side = normalize(cross(cam.dir, make_float3(0, 1, 0)));
	cam.up = normalize(cross(cam.side, cam.dir));
	float fov = 128.0f / 180.0f * float(M_PI);
	cam.halffov = std::tan(fov / 2.0f);

	for (int i = 0; i < FRAMES; i++) {
		float *deviceImage;
		cudaMalloc(&deviceImage, 3 * width * height * sizeof(float));
		
		unsigned long long seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		
		float t = 0.0f;
		if (FRAMES > 1) t = float(i) / (FRAMES - 1.0f);

		render << <blocks, threads >> >(width, height, deviceImage, cam, seed, t);
		
		float *hostImage = (float*)malloc(3 * width * height * sizeof(float));
		cudaMemcpy(hostImage, deviceImage, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost);
		std::string imageName = "renders/render_" + std::to_string(i + 1) + ".png";
		saveImage(imageName, width, height, hostImage);
		cudaFree(deviceImage);
		free(hostImage);

		std::cout << "Frame " << (i + 1) << " done! Saved as " << imageName << "." << std::endl;
	}

	return 0;
}
