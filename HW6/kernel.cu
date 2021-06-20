#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define KERNEL_NUM 0

#define RUN_COUNT 10
#define BLOCK_SIZE_EXP 10

float computeTimeSum = 0, totalTimeSum = 0;

void constantInit(int *data, int size, int val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}

void executeKernel(dim3 gridDim, dim3 blockDim, int *inputData, int *outputData) {
	#if KERNEL_NUM == 1
		matMulA1Kernel <<<gridDim, blockDim>>> (d_C, d_A, d_B, n);
	#elif KERNEL_NUM == 2
		matMulA2Kernel <<<gridDim, blockDim>>> (d_C, d_A, d_B, n);
	#elif KERNEL_NUM == 3
		matMulA3Kernel <<<gridDim, blockDim>>> (d_C, d_A, d_B, n);
	#elif KERNEL_NUM == 4
		matMulA4Kernel <<<gridDim, blockDim>>> (d_C, d_A, d_B, n);
	#elif KERNEL_NUM == 5
		matMulA4Kernel <<<gridDim, blockDim>>> (d_C, d_A, d_B, n);
	#endif
}

void reduce(int exp) {
    int n = 1 << exp;
	size_t mem_size = sizeof(int) * n;

    int rounds = (exp + BLOCK_SIZE_EXP - 1) / BLOCK_SIZE_EXP;

	cudaError_t error;

	// Allocate host memory for matrices A and B
	int *h_A;
    error = cudaMallocHost(&h_A, mem_size);
    if (error != cudaSuccess) {
        printf("[!] A allocation failed!\n");
        exit(1);
    }
	constantInit(h_A, n, 1);

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
	for (int round = rounds; round > 0; round--, exp -= BLOCK_SIZE_EXP, n = 1 << exp) {
		if (round == 1) { // it's the last round
			blockDim = dim3(n, 1, 1);
			gridDim = dim3(1, 1, 1);
		} else {
			blockDim = dim3(BLOCK_SIZE, 1, 1);
			gridDim = dim3(1 << (exp - BLOCK_SIZE_EXP), 1, 1);
		}
		printf("blocks: (%d, %d, %d), grid(%d, %d, %d), exp: %d, n: %d\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, exp, n);

		// Execute the kernel
		outputData = cudaMalloc(&outputData, size);
		executeKernel(gridDim, blockDim, inputData, outputData);
		cudaFree(inputData);
		inputData = outputData;
	}
	
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
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
	// error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	// if (error != cudaSuccess) {
	// 	printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	// 	exit(EXIT_FAILURE);
	// }

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

	printf("Total time = %f, compute time = %f (ms)\n", totalMsec, computeMsec);
	totalTimeSum += totalMsec;
	computeTimeSum += computeMsec;

	// Clean up memory
	cudaFreeHost(h_A);
	cudaFree(dev_A);
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	// Size of square matrices
	unsigned short exp = 0;
	printf("[-] Exponent = ");
	scanf("%u", &exp);
    long n = 1 << exp;

    printf("Reducing %ld size array.\n", n);

	for (int i = 0; i < RUN_COUNT; i++) {
        reduce(exp);
	}

    float avg = computeTimeSum / RUN_COUNT;
	printf("\n[-] Average total time = %f , average compute time = %f\n", totalTimeSum / RUN_COUNT, avg);
    printf("[-] Average memory bandwidth = %f GB/s\n", n * 4.0 / (avg / 1000) / (1 << 30));

	return 0;
}
