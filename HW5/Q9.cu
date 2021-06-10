#define _CRT_SECURE_NO_WARNINGS
#define RUN_COUNT 10

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

double starttime, elapsedtime;
double times_sum = 0;

int *allocateVector(int size);
void fillVector(int * v, size_t n);
void printVector(int * v, size_t n);

void serialAdd(int *a, int *b, int *c, int vectorSize);
void cpuAdd(int *a, int *b, int *c, int vectorSize);
cudaError_t gpuAdd(int *h_a, int *h_b, int *h_c, int vectorSize);

int main()
{
    #ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	const int vectorSize = 100000000;
	int *a, *b, *c;

	srand(time(NULL));

	for (int i = 0; i < RUN_COUNT; i++)
	{
		a = allocateVector(vectorSize);
		b = allocateVector(vectorSize);
		c = allocateVector(vectorSize);
		fillVector(a, vectorSize);
		fillVector(b, vectorSize);
		
        // serialAdd(a, b, c, vectorSize);
        // cpuAdd(a, b, c, vectorSize);
        gpuAdd(a, b, c, vectorSize);

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
    #pragma omp parallel for
	for (i = 0; i < n; i++) {
		v[i] = rand() % 100;
	}
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

void serialAdd(int *a, int *b, int *c, int vectorSize) {
    // get starting time
    starttime = omp_get_wtime();

    int i;
	for (i = 0; i < vectorSize; i++) {
		c[i] = a[i] + b[i];
	}

    // get ending time and use it to determine elapsed time
    elapsedtime = omp_get_wtime() - starttime;
}

void cpuAdd(int *a, int *b, int *c, int vectorSize) {
    // get starting time
    starttime = omp_get_wtime();

    int i;
    #pragma omp parallel for
	for (i = 0; i < vectorSize; i++) {
		c[i] = a[i] + b[i];
	}

    // get ending time and use it to determine elapsed time
    elapsedtime = omp_get_wtime() - starttime;
}

__global__ void addKernel(int *a, int *b, int *c, int vectorSize, int elements_per_thread) {
    int start = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
	for (int i = start; i - start < elements_per_thread && (i < vectorSize); i++) {
		c[i] = a[i] + b[i];
	}
}

cudaError_t gpuAdd(int *h_a, int *h_b, int *h_c, int vectorSize) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	int *dev_a, *dev_b, *dev_c;
	cudaStatus = cudaMalloc(&dev_a, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc(&dev_b, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc(&dev_c, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaMemcpy(dev_a, h_a, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaMemcpy(dev_b, h_b, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	int ELEMENTS_PER_THREAD = 1;
	dim3 NUM_THREADS(512, 1, 1);	// Threads per block
	dim3 NUM_BLOCKS((vectorSize + (NUM_THREADS.x * ELEMENTS_PER_THREAD) - 1) / (NUM_THREADS.x * ELEMENTS_PER_THREAD), 1, 1);

	printf("threads per blocks: %d, blocks: %d\n", NUM_THREADS.x, NUM_BLOCKS.x);

	// get starting time
    starttime = omp_get_wtime();

	// Kernel call
	addKernel<<<NUM_BLOCKS, NUM_THREADS>>>(dev_a, dev_b, dev_c, vectorSize, ELEMENTS_PER_THREAD);
	cudaDeviceSynchronize();

	// get ending time and use it to determine elapsed time
    elapsedtime = omp_get_wtime() - starttime;

	cudaMemcpy(h_c, dev_c, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Check results
	int i;
    #pragma omp parallel for
	for (i = 0; i < vectorSize; i++) {
		if (h_c[i] != h_a[i] + h_b[i]) {
			fprintf(stderr, "wrong addition\n");
			exit(1);
		}
	}

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
