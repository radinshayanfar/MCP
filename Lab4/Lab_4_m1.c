/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#define RUN_COUNT 10

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>

void omp_check();
void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);

int main(int argc, char *argv[]) {
	// Check for correct compilation settings
	omp_check();

	// Input N
	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);

	double starttime, elapsedtime;
	double times_sum = 0;

	for (int i = 0; i < RUN_COUNT; i++)
	{
		// get starting time
		starttime = omp_get_wtime();

		// Allocate memory for array
		int * a = (int *)malloc(n * sizeof a);

		// Fill array with numbers 1..n
		fill_array(a, n);

		// Print array
		// print_array(a, n);

		// Compute prefix sum
		prefix_sum(a, n);

		// Print array
		// print_array(a, n);

		// Free allocated memory
		free(a);

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

void prefix_sum(int *a, size_t n) {
	int *last_sums, *starts, *ends;
	#pragma omp parallel num_threads(8)
	{
		int thread_num = omp_get_thread_num(), num_threads = omp_get_num_threads();
		#pragma omp single
		{
			last_sums 	= (int *) malloc(num_threads * sizeof(int));
			starts 		= (int *) malloc(num_threads * sizeof(int));
			ends 		= (int *) malloc(num_threads * sizeof(int));
		}

		int workload_size = ceil((double)n / num_threads);
		starts[thread_num] = thread_num * workload_size;
		ends[thread_num] = starts[thread_num] + workload_size;
		int excessive = workload_size * num_threads - n;
		if (thread_num >= num_threads - excessive) {
			starts[thread_num] 	-= thread_num - num_threads + excessive;
			ends[thread_num] 	-= thread_num - num_threads + excessive + 1;
		}
		// printf("th: %d, start: %d, end: %d\n", thread_num, starts[thread_num], ends[thread_num]);

		for (int i = starts[thread_num] + 1; i < ends[thread_num]; i++) {
			a[i] += a[i - 1];
		}
		#pragma omp barrier

		// #pragma omp single
		// 	print_array(a, n);
		
		#pragma omp single
		{
			// starts[0] = 1;						// just to avoid deadlock by putting barrier inside if
			last_sums[0] = 0;
			for (int i = 1; i < num_threads; i++) {
				last_sums[i] = a[starts[i] - 1] + last_sums[i - 1];
			}
		}
		
		int const_sum = last_sums[thread_num];
		if (thread_num != 0) {
			// printf("th: %d, start: %d, end: %d, const: %d\n", thread_num, starts[thread_num], ends[thread_num], const_sum);
			for (int i = starts[thread_num]; i < ends[thread_num]; i++) {
				a[i] += const_sum;
			}
		}
		// int i;
		// // #pragma omp for private(i)
		// for (i = starts[thread_num]; i < end[thread_num]; i++) {
		// 	a[i] = omp_get_thread_num();
		// }
		
	}
}

void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%3d, ", a[i]);
	}
	printf("\b\b  \n");
}

void fill_array(int *a, size_t n) {
	int i;
	#pragma omp parallel for
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}

void omp_check() {
	printf("------------ Info -------------\n");
#ifdef _DEBUG
	printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
	printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
	printf("[-] Platform: x64\n");
#elif _M_IX86 
	printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
	printf("[-] OpenMP is on.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
	printf("[!] OpenMP is off.\n");
	printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
	printf("[-] Maximum threads: %d\n", omp_get_max_threads());
	printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}
