/*
*	In His Exalted Name
*	Vector Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	21/05/2018
*/

#define RUN_COUNT 10

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double starttime, elapsedtime;
double times_sum = 0;

int *allocateVector(int size);
void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main()
{
	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	const int vectorSize = 1 << 26;
	int *a, *b, *c;

	for (int i = 0; i < RUN_COUNT; i++)
	{
		a = allocateVector(vectorSize);
		b = allocateVector(vectorSize);
		c = allocateVector(vectorSize);
		fillVector(a, vectorSize);
		fillVector(b, vectorSize);

		// addVector(a, b, c, vectorSize);
		addWithCuda(c, a, b, vectorSize);

		// printVector(c, vectorSize);

		free(a);
        free(b);
        free(c);

		// report elapsed time
		printf("[-] Time Elapsed: %f Secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf\n", times_sum / RUN_COUNT);	

	return EXIT_SUCCESS;
}

// Allocates vector in host 
int *allocateVector(int size) {
	return (int *) malloc(sizeof(int) * size);
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	// get starting time
	starttime = omp_get_wtime();

	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}

	// get starting time
	starttime = omp_get_wtime();
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}

__global__ void addKernel(int *c, const int *a, const int *b, const int vectorSize, const int elements_per_thread)
{
    int start = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
	for (int i = start; i - start < elements_per_thread && (i < vectorSize); i++) {
		c[i] = a[i] + b[i];
	}
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	int ELEMENTS_PER_THREAD = 1;
	dim3 NUM_THREADS(1024, 1, 1);	// Threads per block
	dim3 NUM_BLOCKS((size + (NUM_THREADS.x * ELEMENTS_PER_THREAD) - 1) / (NUM_THREADS.x * ELEMENTS_PER_THREAD), 1, 1);

	printf("elements per thread: %d, threads per blocks: %d, blocks: %d\n", ELEMENTS_PER_THREAD, NUM_THREADS.x, NUM_BLOCKS.x);

	// get starting time
	starttime = omp_get_wtime();

	// Launch a kernel on the GPU with one thread for each element.
    addKernel<<<NUM_BLOCKS, NUM_THREADS>>>(dev_c, dev_a, dev_b, size, ELEMENTS_PER_THREAD);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// get ending time and use it to determine elapsed time
	elapsedtime = omp_get_wtime() - starttime;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
