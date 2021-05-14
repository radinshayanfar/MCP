#include <iostream>
#include "omp.h"
#include <cmath>

int main(int argc, char* argv) {

	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	int E = 4;
	omp_set_num_threads(4);
	int arr_size = 929;
	int* arr = (int*)malloc(sizeof(int) * arr_size);
	#pragma omp parallel
	{
		int workload_size = ceil((double)arr_size / omp_get_num_threads());
		int start = omp_get_thread_num() * workload_size;
		int end = start + workload_size;

		printf("thread: %d, start: %d, end: %d\n", omp_get_thread_num(), start, end);
			
		for (int i = start; (i < end) && (i < arr_size); i++)
		{
			arr[i] = 0;
		}
	}

	return 0;
}