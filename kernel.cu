#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "lodepng.h"

#define BLOCK_SIZE 8

struct Hit 
{
	bool isHit = 0;
	float3 pos;
	float3 normal;
	float3 color;
};

__device__ Hit march() 
{
	Hit test;
	test.color = make_float3(1.0f, 0.0f, 1.0f);
	test.isHit = 1;
	return test;
}

__global__ void trace(int width, int height, float* result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	Hit rayhit = march();
	
	float3 color = (rayhit.isHit) ? rayhit.color : make_float3(0, 0, 0);

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

	trace << <blocks, threads >> >(width, height, deviceImage);

	float *hostImage = (float*) malloc(3 * width * height * sizeof(float));
	cudaMemcpy(hostImage, deviceImage, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost);
	saveImage(width, height, hostImage);
	cudaFree(deviceImage);
	free(hostImage);
    return 0;
}
