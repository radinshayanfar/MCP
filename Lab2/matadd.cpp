/*
*	In His Exalted Name
*	Matrix Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	15/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#define RUN_COUNT 1

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>

typedef struct {
	int *A, *B, *C;
	int n, m;
} DataSet;

void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void add(DataSet dataSet);
void add2D(DataSet dataSet);

int main(int argc, char *argv[]) {
	DataSet dataSet;
	if (argc < 3) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> \n");
		printf(">>> ");
		scanf("%d %d", &dataSet.n, &dataSet.m);
	}
	else {
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
	}

	printf("[-] Dataset size is: %d bytes\n\n", dataSet.n * dataSet.m * sizeof(int));

	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif
	// omp_set_num_threads(8);

	double starttime, elapsedtime;
	double times_sum = 0;

	for (int i = 0; i < RUN_COUNT; i++)
	{
		// get starting time
		starttime = omp_get_wtime();

		fillDataSet(&dataSet);
		add2D(dataSet);
		printDataSet(dataSet);
		closeDataSet(dataSet);

		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;
		// report elapsed time
		printf("[-] Time Elapsed: %f Secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf\n", times_sum / RUN_COUNT);	
	
	system("PAUSE");
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet) {
	int i, j;

	dataSet->A = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->B = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->C = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);

	srand(time(NULL));

	// #pragma omp parallel for num_threads(8)
	for (i = 0; i < dataSet->n; i++) {
		for (j = 0; j < dataSet->m; j++) {
			dataSet->A[i*dataSet->m + j] = rand() % 100;
			dataSet->B[i*dataSet->m + j] = rand() % 100;
		}
	}
}

void printDataSet(DataSet dataSet) {
	int i, j;

	printf("[-] Matrix A\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.A[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix B\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.B[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.C[i*dataSet.m + j]);
		}
		putchar('\n');
	}
}

void closeDataSet(DataSet dataSet) {
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}

void add(DataSet dataSet) {
	int i, j;
	#pragma omp parallel for num_threads(2)
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
			// dataSet.C[i * dataSet.m + j] = omp_get_thread_num();
		}
	}
}

void add2D(DataSet dataSet) {
	#pragma omp parallel num_threads(16)
	{
		int i, j;

		int n_threads = omp_get_num_threads();
		int row_blocks = sqrt(n_threads);
		int col_blocks = sqrt(n_threads);

		int row_block_size = dataSet.n / row_blocks;
		int col_block_size = dataSet.m / col_blocks;

		int curr_thread = omp_get_thread_num();

		int row = curr_thread / row_blocks;
		int col = curr_thread % col_blocks;

		for (i = row * row_block_size; i < (row + 1) * row_block_size; i++) {
			for (j = col * col_block_size; j < (col + 1) * col_block_size; j++) {
				dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
				// dataSet.C[i * dataSet.m + j] = omp_get_thread_num();
			}
		}
	}
}
