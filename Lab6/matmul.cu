// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

// #define DEBUG
#define APPROACH 4

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RUN_COUNT 5
double timesSum = 0;


/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
#define BLOCK_WIDTH 32

__global__ void matMulA1Kernel(float *C, float *A, float *B, int n) {
	int col_workload = (n + blockDim.x - 1) / blockDim.x;
	int row_workload = (n + blockDim.y - 1) / blockDim.y;

	int col_start = col_workload * threadIdx.x;
	int col_end = col_start + col_workload;
	int row_start = row_workload * threadIdx.y;
	int row_end = row_start + row_workload;

	float sum;

	for (int row = row_start; (row < row_end) && (row < n); row++) {
		for (int col = col_start; (col < col_end) && (col < n); col++) {
			sum = 0;
			for (int k = 0; k < n; k++) {
				sum += A[row * n + k] * B[k * n + col];
			}
			C[row * n + col] = sum;
		}
	}
	
}

__global__ void matMulA2Kernel(float *C, float *A, float *B, int n) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	float sum = 0.0f;

	if (col >= n || row >= n) {
		return;
	}

	for (int k = 0; k < n; k++) {
		sum += A[row * n + k] * B[k * n + col];
	}
	C[row * n + col] = sum;
}

__global__ void matMulA3Kernel(float *C, float *A, float *B, int n) {
	int TILE_WIDTH = blockDim.x;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	float sum = 0.0f;

	if (col >= n || row >= n) {
		return;
	}

	int tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

	for (int stage = 0; stage < tiles; stage++) {
		for (int k = 0; (k < TILE_WIDTH) && (stage * TILE_WIDTH + k < n); k++) {
			sum += A[row * n + (stage * TILE_WIDTH + k)] * B[(stage * TILE_WIDTH + k) * n + col];
		}
		C[row * n + col] = sum;
	}
}

__global__ void matMulA4Kernel(float *C, float *A, float *B, int n) {
	__shared__ int s_A[BLOCK_WIDTH][BLOCK_WIDTH];
	__shared__ int s_B[BLOCK_WIDTH][BLOCK_WIDTH];

	int TILE_WIDTH = blockDim.x;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	float sum = 0.0f;

	if (col >= n || row >= n) {
		return;
	}

	int tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

	for (int stage = 0; stage < tiles; stage++) {
		s_A[threadIdx.y][threadIdx.x] = A[(row) * n + (stage * TILE_WIDTH + threadIdx.x)];
		s_B[threadIdx.y][threadIdx.x] = B[(stage * TILE_WIDTH + threadIdx.y) * n + (col)];

		__syncthreads();

		for (int k = 0; (k < TILE_WIDTH) && (stage * TILE_WIDTH + k < n); k++) {
			sum += A[row * n + k] * B[k * n + col];
		}

		__syncthreads();
	}

	C[row * n + col] = sum;
}

void constantInit(float *data, int size, float val) {
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int argc, char **argv, int n) {
	// Allocate host memory for matrices A and B
	unsigned int size_A = n * n;
	unsigned int mem_size_A = sizeof(float)* size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = n * n;
	unsigned int mem_size_B = sizeof(float)* size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	unsigned int mem_size_C = n * n * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters

	#if APPROACH == 1
		dim3 threads(BLOCK_WIDTH, BLOCK_WIDTH, 1);
		dim3 grid(1, 1, 1);
	#else
		dim3 threads(BLOCK_WIDTH, BLOCK_WIDTH, 1);
		int gridOneDim = (n + threads.x - 1) / threads.x;
		dim3 grid(gridOneDim, gridOneDim, 1);
	#endif

	#ifdef DEBUG
		printf("threads: (%d, %d, %d), blocks(%d, %d, %d)\n", threads.x, threads.y, threads.z, grid.x, grid.y, grid.z);
	#endif

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	#if APPROACH == 1
		matMulA1Kernel <<<grid, threads>>> (d_C, d_A, d_B, n);
	#elif APPROACH == 2
		matMulA2Kernel <<<grid, threads>>> (d_C, d_A, d_B, n);
	#elif APPROACH == 3
		matMulA3Kernel <<<grid, threads>>> (d_C, d_A, d_B, n);
	#elif APPROACH == 4
		matMulA4Kernel <<<grid, threads>>> (d_C, d_A, d_B, n);
	#endif
	

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	printf("Elapsed time in msec = %f\n", msecTotal);
	timesSum += msecTotal;

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	#ifdef DEBUG
		// Checking results on CPU
		float sum;
		for (int i = 0; i < n * n; i++) {
			sum = 0;
			for (int k = 0; k < n; k++)
			{
				sum += h_A[(i / n) * n + k] * h_B[k * n + (i % n)];
			}
			if (h_C[(i / n) * n + (i % n)] != sum) {
				fprintf(stderr, "wrong answer\n");
				exit(1);
			}
		}
	#endif


	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;

}

void matMulSerial(int argc, char **argv, int n) {
	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return;
	#endif

	// Allocate host memory for matrices A and B
	unsigned int size_A = n * n;
	unsigned int mem_size_A = sizeof(float)* size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = n * n;
	unsigned int mem_size_B = sizeof(float)* size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);

	// Allocate host matrix C
	unsigned int mem_size_C = n * n * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	// get starting time
	double starttime = omp_get_wtime();

	float sum;
	for (int i = 0; i < n * n; i++) {
		sum = 0;
		for (int k = 0; k < n; k++)
		{
			sum += h_A[(i / n) * n + k] * h_B[k * n + (i % n)];
		}
		h_C[(i / n) * n + (i % n)] = sum;
	}

	// get ending time and use it to determine elapsed time
	double elapsedtime = (omp_get_wtime() - starttime) * 1000;

	printf("Elapsed time in msec = %lf\n", elapsedtime);
	timesSum += elapsedtime;

	free(h_A);
 	free(h_B);
 	free(h_C);
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	// By default, we use device 0
	int devID = 0;
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Size of square matrices
	size_t n = 0;
	printf("[-] N = ");
	scanf("%u", &n);

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", n, n, n, n);

	#if APPROACH == 0
		printf("Computing with approach 0 (serial)\n");
	#elif APPROACH == 1
		printf("Computing with approach 1\n");
	#elif APPROACH == 2
		printf("Computing with approach 2\n");
	#elif APPROACH == 3
		printf("Computing with approach 3\n");
	#elif APPROACH == 4
		printf("Computing with approach 4\n");
	#endif

	for (int i = 0; i < RUN_COUNT; i++) {
		#if APPROACH == 0
			matMulSerial(argc, argv, n);
		#else
			matrixMultiply(argc, argv, n);
		#endif
	}
	printf("\nAverage time in msec = %f\n", timesSum / RUN_COUNT);

	return 0;
}
