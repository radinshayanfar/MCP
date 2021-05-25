#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <cmath>

#define RUN_COUNT 1

typedef struct {
	float **A, **L, **U;
	int n, blocks  = 1;
} Dataset;

void fillDataset(Dataset *dataset);
// void printDataset(Dataset dataset);
void printMatrix(int n, float **mat);
void closeDataset(Dataset dataset);
void luDecomposition(Dataset dataset, int row, int column);

int main(int argc, char* argv[]) {
	Dataset dataset;
	if (argc < 2) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> \n");
		printf(">>> ");
		scanf("%d", &(dataset.n));
	}
	else {
		dataset.n = atoi(argv[1]);
	}

	printf("[-] dim size is: %d, and dataset size is: %lu bytes\n\n", dataset.n, (unsigned int)pow(dataset.n, 2) * sizeof(int));

	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif
	omp_set_num_threads(4);

	double starttime, elapsedtime;
	double times_sum = 0;

	for (int i = 0; i < RUN_COUNT; i++)
	{
		fillDataset(&dataset);
	
		// get starting time
		starttime = omp_get_wtime();

		printMatrix(dataset.n, dataset.A);
		luDecomposition(dataset, 0, 0);
		printMatrix(dataset.n, dataset.A);

		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;

		// printDataset(dataset);
		closeDataset(dataset);

		// report elapsed time
		printf("[-] Time Elapsed: %f Secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf secs.\n", times_sum / RUN_COUNT);

	return 0;
}

float **allocateMatrix(int n) {
	float **mat = (float**)malloc(sizeof(float*) * n);
	for (int i = 0; i < n; i++)
	{
		mat[i] = (float*)malloc(sizeof(float) * n);
	}
	return mat;
}

void fillDataset(Dataset* dataset) {
	int n = dataset->n;
	dataset->A = allocateMatrix(n);
	dataset->L = allocateMatrix(n);
	dataset->U = allocateMatrix(n);
	
	srand(time(NULL));

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			dataset->A[i][j] = rand() % 30;
			dataset->L[i][j] = 0;
			dataset->U[i][j] = 0;
		}
	}
}

void printMatrix(int n, float **mat) {
	// printf("[");
	for (int i = 0; i < n; i++) {
		// printf("[");
		for (int j = 0; j < n; j++) {
			// printf("%.2f, ", mat[i][j]);
			printf("%5.2f ", mat[i][j]);
		}
		// printf("], \n");
		printf("\n");
	}
	// printf("]\n");
}

void closeDataset(Dataset dataset) {
	for (int i = 0; i < dataset.n; i++)
	{
		free(dataset.A[i]);
		free(dataset.L[i]);
		free(dataset.U[i]);
	}
	free(dataset.A);
	free(dataset.L);
	free(dataset.U);
}

void luDecomposition(Dataset dataset, int row, int column) {
	int n = dataset.n;
	int blocks = dataset.blocks;
	float **A = dataset.A, **L = dataset.L, **U = dataset.U;

	int blocksize = n / blocks;
	int row_start 		= row 		* blocksize	, row_stop 		= (row + 1) 	* blocksize;
	int column_start 	= column 	* blocksize	, column_stop 	= (column + 1) 	* blocksize;

	float *temp = (float *)malloc(sizeof(float) * blocksize);

	for (int k = row_start; k < row_stop; k++) {
		for (int j = k; j < column_stop; j++) {
			temp[j] = A[k][j] / A[k][k];					// temp[j] == A[k][j]
		}
		for (int i = k + 1; i < row_stop; i++) {
			for (int j = k + 1; j < column_stop; j++) {
				A[i][j] -= A[i][k] * temp[j];
			}
			A[i][k] /= A[k][k];
		}
	}
}