#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <time.h>
#include <omp.h>
#include <cmath>

#define RUN_COUNT 1

typedef struct {
	unsigned short ***A, ***B, ***C;
	int n;
} Dataset;

unsigned short *** fillDataset(Dataset *dataset);
void printDataset(Dataset dataset);
void closeDataset(Dataset dataset);
void mulRow(Dataset dataset);
void mulCol(Dataset dataset);
void mulBlock(Dataset dataset);

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
	
	printf("[-] dim size is: %d, and dataset size is: %u bytes\n\n", dataset.n, (unsigned int)pow(dataset.n, 3) * sizeof(unsigned short));

	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	double starttime, elapsedtime;
	double times_sum = 0;

	for (int i = 0; i < RUN_COUNT; i++)
	{
		fillDataset(&dataset);

		// get starting time
		starttime = omp_get_wtime();

		mulBlock(dataset);
		// mulRow(dataset);
		// mulCol(dataset);

		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;

		// printDataset(dataset);
		closeDataset(dataset);

		// report elapsed time
		printf("[-] Time Elapsed: %f Secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf\n", times_sum / RUN_COUNT);

	system("PAUSE");
	return 0;
}

unsigned short*** fillDataset(Dataset *dataset) {
	dataset->A = (unsigned short***)malloc(sizeof(unsigned short**) * dataset->n);
	dataset->B = (unsigned short***)malloc(sizeof(unsigned short**) * dataset->n);
	dataset->C = (unsigned short***)malloc(sizeof(unsigned short**) * dataset->n);
	for (int i = 0; i < dataset->n; i++)
	{
		dataset->A[i] = (unsigned short**)malloc(sizeof(unsigned short*) * dataset->n);
		dataset->B[i] = (unsigned short**)malloc(sizeof(unsigned short*) * dataset->n);
		dataset->C[i] = (unsigned short**)malloc(sizeof(unsigned short*) * dataset->n);
	}
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < dataset->n; i++)
	{
		for (int j = 0; j < dataset->n; j++)
		{
			dataset->A[i][j] = (unsigned short*)malloc(sizeof(unsigned short) * dataset->n);
			dataset->B[i][j] = (unsigned short*)malloc(sizeof(unsigned short) * dataset->n);
			dataset->C[i][j] = (unsigned short*)malloc(sizeof(unsigned short) * dataset->n);
		}
	}

	srand(time(NULL));

	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < dataset->n; i++) {
		for (int j = 0; j < dataset->n; j++) {
			for (int k = 0; k < dataset->n; k++)
			{
				dataset->A[i][j][k] = rand() % 100;
				dataset->B[i][j][k] = rand() % 100;
				dataset->C[i][j][k] = 0;
			}
		}
	}

	return NULL;
}

void mulRow(Dataset dataset) {
	for (int i = 0; i < dataset.n; i++) {
		#pragma omp parallel for num_threads(16)
		for (int j = 0; j < dataset.n; j++)  {
			for (int k = 0; k < dataset.n; k++) {
				for (int l = 0; l < dataset.n; l++) {
					dataset.C[i][j][k] += dataset.A[i][j][l] * dataset.B[i][l][k];
					// dataset.C[i][j][k] = omp_get_thread_num();
				}
			}
		}
	}
}

void mulCol(Dataset dataset) {
	for (int i = 0; i < dataset.n; i++) {
		for (int j = 0; j < dataset.n; j++)  {
			#pragma omp parallel for num_threads(16)
			for (int k = 0; k < dataset.n; k++) {
				for (int l = 0; l < dataset.n; l++) {
					dataset.C[i][j][k] += dataset.A[i][j][l] * dataset.B[i][l][k];
					// dataset.C[i][j][k] = omp_get_thread_num();
				}
			}
		}
	}
}

void mulBlock(Dataset dataset) {
	#pragma omp parallel for num_threads(16)
	for (int i = 0; i < dataset.n; i++) {
		for (int j = 0; j < dataset.n; j++)  {
			for (int k = 0; k < dataset.n; k++) {
				for (int l = 0; l < dataset.n; l++) {
					dataset.C[i][j][k] += dataset.A[i][j][l] * dataset.B[i][l][k];
					// dataset.C[i][j][k] = omp_get_thread_num();
				}
			}
		}
	}
}

void print3DMatrix(unsigned short ***mat, int size) {
	printf("[");
	for (int i = 0; i < size; i++) {
		printf("[\n");
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				printf("%d ", mat[i][j][k]);
			}
			printf("\n");
		}
		printf("]");
	}
	printf("]\n");
}

void printDataset(Dataset dataset) {
	printf("[-] Matrix A\n");
	print3DMatrix(dataset.A, dataset.n);

	printf("[-] Matrix B\n");
	print3DMatrix(dataset.B, dataset.n);

	printf("[-] Matrix C\n");
	print3DMatrix(dataset.C, dataset.n);
}

void closeDataset(Dataset dataset) {
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < dataset.n; i++)
	{
		for (int j = 0; j < dataset.n; j++)
		{
			free(dataset.A[i][j]);
			free(dataset.B[i][j]);
			free(dataset.C[i][j]);
		}
	}
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
