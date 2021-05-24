#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <omp.h>
#include <cmath>

#define RUN_COUNT 10

typedef struct {
	int **A, **B, **C, **R;
	int n;
} Dataset;

void fillDataset(Dataset *dataset);
// void printDataset(Dataset dataset);
void printMatrix(int n, int **mat);
void closeDataset(Dataset dataset);
void add(int n, int **A, int **B, int **C);
void multiply(int n, int **A, int **B, int **C);
void transpose(int n, int **A, int **B);
void inPlaceTranspose(int n, int **A);
void compute(Dataset dataset);

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

	printf("[-] dim size is: %d, and dataset size is: %u bytes\n\n", dataset.n, (unsigned int)pow(dataset.n, 2) * sizeof(int));

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

		compute(dataset);

		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;

		// printDataset(dataset);
		closeDataset(dataset);

		// report elapsed time
		printf("[-] Time Elapsed: %f Secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf secs.\n", times_sum / RUN_COUNT);

	system("PAUSE");
	return 0;
}

int **allocateMatrix(int n) {
	int **mat = (int**)malloc(sizeof(int*) * n);
	for (int i = 0; i < n; i++)
	{
		mat[i] = (int*)malloc(sizeof(int) * n);
	}
	return mat;
}

void fillDataset(Dataset* dataset) {
	int n = dataset->n;
	dataset->A = allocateMatrix(n);
	dataset->B = allocateMatrix(n);
	dataset->C = allocateMatrix(n);
	dataset->R = allocateMatrix(n);
	
	srand(time(NULL));

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			dataset->A[i][j] = rand() % 30;
			dataset->B[i][j] = rand() % 30;
			dataset->C[i][j] = rand() % 30;
			dataset->R[i][j] = 0;
		}
	}
}

void printMatrix(int n, int **mat) {
	printf("[");
	for (int i = 0; i < n; i++) {
		printf("[");
		for (int j = 0; j < n; j++) {
			printf("%d, ", mat[i][j]);
		}
		printf("], \n");
	}
	printf("]\n");
}

void closeDataset(Dataset dataset) {
	for (int i = 0; i < dataset.n; i++)
	{
		free(dataset.A[i]);
		free(dataset.B[i]);
		free(dataset.C[i]);
	}
	free(dataset.A);
	free(dataset.B);
	free(dataset.C);
}

void add(int n, int **A, int **B, int **C) {
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
}

void multiply(int n, int **A, int **B, int **C) {
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			C[i][j] = 0;
			for (int k = 0; k < n; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void transpose(int n, int **A, int **B) {
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			B[j][i] = A[i][j];
		}
	}
}

void inPlaceTranspose(int n, int **A) {
	int t;
	// DON'T FORGET private(t) !!!!!
	#pragma omp parallel for private(t) schedule(static, n / 32)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			// swapping A[i][j] <==> A[j][i]
			t = A[i][j];
			A[i][j] = A[j][i];
			A[j][i] = t;
		}
	}
}

void compute(Dataset dataset) {
	int n = dataset.n;

	int **A_T = allocateMatrix(n);
	int **AT_A = allocateMatrix(n);
	int **B_A = allocateMatrix(n);

	transpose(n, dataset.A, A_T);
	multiply(n, A_T, dataset.A, AT_A);
	multiply(n, dataset.B, dataset.A, B_A);
	add(n, AT_A, B_A, dataset.A);				// A contains the sum
	inPlaceTranspose(n, dataset.C);				// C contains C^T
	multiply(n, dataset.A, dataset.C, dataset.R);
}
