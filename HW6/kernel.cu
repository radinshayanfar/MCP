#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// #define DEBUG
#define KERNEL_NUM 5

#define RUN_COUNT 10
#define BLOCK_SIZE_EXP 8

float computeTimeSum = 0, totalTimeSum = 0;

__global__ void kernel1(int *inputData, int *outputData) {
	extern __shared__ int sdata[];

	// each threads loads one element from global to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = inputData[i];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global memory
	if (tid == 0) {
		outputData[blockIdx.x] = sdata[0];
	}
}

__global__ void kernel2(int *inputData, int *outputData) {
	extern __shared__ int sdata[];

	// each threads loads one element from global to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = inputData[i];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}

		__syncthreads();
	}

	// write result for this block to global memory
	if (tid == 0) {
		outputData[blockIdx.x] = sdata[0];
	}
}

__global__ void kernel3(int *inputData, int *outputData) {
	extern __shared__ int sdata[];

	// each threads loads one element from global to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = inputData[i];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}

	// write result for this block to global memory
	if (tid == 0) {
		outputData[blockIdx.x] = sdata[0];
	}
}

__global__ void kernel4(int *inputData, int *outputData) {
	extern __shared__ int sdata[];

	// each threads loads one element from global to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = inputData[i] + inputData[i + blockDim.x];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}

	// write result for this block to global memory
	if (tid == 0) {
		outputData[blockIdx.x] = sdata[0];
		// printf("outputData[%d]: %d\n", blockIdx.x, outputData[blockIdx.x]);
	}
}

__global__ void wrongKernel5(int *inputData, int *outputData) {
	extern __shared__ int sdata[];

	// each threads loads one element from global to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = inputData[i] + inputData[i + blockDim.x];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}

	if (tid < 32) {
		if (blockDim.x > 32) {
			sdata[tid] += sdata[tid + 32];
			__syncthreads();
		}
		if (blockDim.x > 16) {
			sdata[tid] += sdata[tid + 16];
			__syncthreads();
		}
		if (blockDim.x > 8) {
			sdata[tid] += sdata[tid + 8];
			__syncthreads();
		}
		if (blockDim.x > 4) {
			sdata[tid] += sdata[tid + 4];
			__syncthreads();
		}
		if (blockDim.x > 2) {
			sdata[tid] += sdata[tid + 2];
			__syncthreads();
		}
		sdata[tid] += sdata[tid + 1];
		__syncthreads();
	}
	

	// write result for this block to global memory
	if (tid == 0) {
		outputData[blockIdx.x] = sdata[0];
		// printf("outputData[%d]: %d\n", blockIdx.x, outputData[blockIdx.x]);
	}
	// if (tid == 0 && blockIdx.x == 0) {
	// 	printf("outputData[%d]: %d\n", blockIdx.x, outputData[blockIdx.x]);
	// }
}

__global__ void kernel5(int *inputData, int *outputData) {
	extern __shared__ int sdata[];

	// each threads loads one element from global to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = inputData[i] + inputData[i + blockDim.x];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		} else {
			return;
		}

		__syncthreads();
	}

	// write result for this block to global memory
	if (tid == 0) {
		outputData[blockIdx.x] = sdata[0];
		// printf("outputData[%d]: %d\n", blockIdx.x, outputData[blockIdx.x]);
	}
}

void constantInit(int *data, int size, int val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}

void randomInit(int *data, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		data[i] = rand() % 10;
	}
}

int checkCPU(int *inputData, int length) {
	int sum = 0;
	#pragma omp parallel for reduction(+: sum)
	for (int i = 0; i < length; i++) {
		sum += inputData[i];
	}
	return sum;
}

void executeKernel(dim3 gridDim, dim3 blockDim, int *inputData, int *outputData) {
	size_t sharedMemorySize = blockDim.x * sizeof(int);
	#if KERNEL_NUM == 1
		kernel1 <<<gridDim, blockDim, sharedMemorySize>>> (inputData, outputData);
	#elif KERNEL_NUM == 2
		kernel2 <<<gridDim, blockDim, sharedMemorySize>>> (inputData, outputData);
	#elif KERNEL_NUM == 3
		kernel3 <<<gridDim, blockDim, sharedMemorySize>>> (inputData, outputData);
	#elif KERNEL_NUM == 4
		kernel4 <<<gridDim, blockDim, sharedMemorySize>>> (inputData, outputData);
	#elif KERNEL_NUM == 5
		kernel5 <<<gridDim, blockDim, sharedMemorySize>>> (inputData, outputData);
	#endif
}

void reduce(const int exp) {
    int n = 1 << exp;
	size_t mem_size = sizeof(int) * n;

	bool halveBlocks = KERNEL_NUM >= 4;

    int rounds = (exp + BLOCK_SIZE_EXP + halveBlocks - 1) / (BLOCK_SIZE_EXP + halveBlocks);

	cudaError_t error;

	// Allocate host memory for matrices A and B
	int *h_A;
    error = cudaMallocHost(&h_A, mem_size);
    if (error != cudaSuccess) {
        printf("[!] A allocation failed!\n");
        exit(1);
    }
	constantInit(h_A, n, 1);
	// randomInit(h_A, n);

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t copyStart, copyStop, computeStart, computeStop;
	error = cudaEventCreate(&copyStart);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to create copyStart event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaEventCreate(&copyStop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to create copyStop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaEventCreate(&computeStart);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to create computeStart event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaEventCreate(&computeStop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to create computeStop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaEventRecord(copyStart, NULL);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to record copyStart event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Allocate device memory
	int *dev_A;
	error = cudaMalloc((void **)&dev_A, mem_size);
	if (error != cudaSuccess) {
		printf("cudaMalloc dev_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(dev_A, h_A, mem_size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("cudaMemcpy (dev_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaEventRecord(computeStart, NULL);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to record computeStart event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	const int BLOCK_SIZE = 1 << BLOCK_SIZE_EXP;
	dim3 gridDim, blockDim;
	int *inputData = dev_A, *outputData;
	int exp_copy = exp;
	for (int round = rounds; round > 0; round--, exp_copy -= (BLOCK_SIZE_EXP + halveBlocks), n = 1 << exp_copy) {
		if (round == 1) {				// it's the last round
			blockDim = dim3(n >> halveBlocks, 1, 1);
			gridDim = dim3(1, 1, 1);
		} else {
			blockDim = dim3(BLOCK_SIZE, 1, 1);
			gridDim = dim3(1 << (exp_copy - BLOCK_SIZE_EXP - halveBlocks), 1, 1);
		}
		printf("blocks: (%d, %d, %d), grid(%d, %d, %d), exp: %d, n: %d\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, exp_copy, n);

		// Execute the kernel
		error = cudaMalloc(&outputData, gridDim.x * sizeof(int));
		if (error != cudaSuccess) {
			printf("cudaMalloc outputData returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
			exit(EXIT_FAILURE);
		}

		executeKernel(gridDim, blockDim, inputData, outputData);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel (%s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to synchronize devices (%s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		error = cudaFree(inputData);
		if (error != cudaSuccess) {
			fprintf(stderr, "Failed to free up inputData (%s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		inputData = outputData;

		// error = cudaMemcpy(h_A, inputData, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);
		// for (int j = 0; j < gridDim.x; j++)
		// 	printf("inputData[%d]: %d\n", j, h_A[j]);
	}
	
	error = cudaEventRecord(computeStop, NULL);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to record computeStop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(computeStop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize on the computeStop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	int *result = (int *) malloc(sizeof(int));
	error = cudaMemcpy(result, inputData, sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaEventRecord(copyStop, NULL);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to record copyStop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaEventSynchronize(copyStop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize on the copyStop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float totalMsec, computeMsec;
	error = cudaEventElapsedTime(&totalMsec, copyStart, copyStop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to get time elapsed between copy events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaEventElapsedTime(&computeMsec, computeStart, computeStop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to get time elapsed between compute events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	printf("Sum = %d, total time = %f, compute time = %f (ms)\n", *result, totalMsec, computeMsec);
	totalTimeSum += totalMsec;
	computeTimeSum += computeMsec;

	#ifdef DEBUG
	// Check results on CPU
	printf("[-] Checking on CPU...");
	int cpuResult = checkCPU(h_A, 1 << exp);
	if (*result != cpuResult) {
		printf("\n[!] Reduction result is not same as CPU's\n");
		printf("[-] GPU: %d, CPU: %d\n", *result, cpuResult);
		exit(1);
	} else {
		printf(" [OK]\n");
	}
	#endif

	// Clean up memory
	error = cudaFreeHost(h_A);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to free up host memory (%s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	error = cudaFree(inputData);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to free up inputData (%s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return;
	#endif
	srand(time(NULL));

	// Size of square matrices
	unsigned short exp = 0;
	printf("[-] Exponent = ");
	scanf("%u", &exp);
    long n = 1 << exp;

    printf("Reducing %ld size array.\n\n", n);

	for (int i = 0; i < RUN_COUNT; i++) {
        reduce(exp);
		printf("\n");
	}

    float avg = computeTimeSum / RUN_COUNT;
	printf("\n[-] Average total time = %f , average compute time = %f\n", totalTimeSum / RUN_COUNT, avg);
    printf("[-] Average memory bandwidth = %f GB/s\n", n * 4.0 / (avg / 1000) / (1 << 30));

	return 0;
}
