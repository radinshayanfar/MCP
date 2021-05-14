#include <iostream>
#include "omp.h"

#define RUN_COUNT 10

int main(int argc, char* argv) {

	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	double starttime, elapsedtime;
	double times_sum = 0;

	int E[300][300];

	for (int i = 0; i < RUN_COUNT; i++)
	{
		starttime = omp_get_wtime();

		#pragma omp parallel for
		for (int i = 0; i < 250; i++)
		{
			for (int j = 0; j < 250; j++)
			{
				E[i][j] += j;
			}
		}

		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;

		// report elapsed time
		printf("[-] Time Elapsed: %f secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf\n", times_sum / RUN_COUNT);

	return 0;
}