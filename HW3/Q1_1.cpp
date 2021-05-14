#include <iostream>
#include "omp.h"

#define RUN_COUNT 10

int main(int argc, char *argv) {

	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	double starttime, elapsedtime;
	double times_sum = 0;

	int i, j, acc;
	for (int i = 0; i < RUN_COUNT; i++)
	{
		starttime = omp_get_wtime();

		acc = 0;
		#pragma omp parallel reduction(+: acc)
		{
			#pragma omp for
			for (j = 0; j < 100; j++)
			{
				acc++;
			}
		}

		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;

		// report elapsed time
		printf("[-] Time Elapsed: %f secs, acc: %d\n", elapsedtime, acc);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf\n", times_sum / RUN_COUNT);

	return 0;
}