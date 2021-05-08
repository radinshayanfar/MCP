
#include <stdio.h>
#include <math.h>
#include <omp.h>

const long int VERYBIG = 50000;
// ***********************************************************************
int main(void)
{
#ifndef _OPENMP
	printf("OpenMP is not supported.\n");
	return 0;
#endif

	int i;
	long int j, k, sum;
	double sumx, sumy, total;
	double starttime, elapsedtime;
	double times_sum = 0;
	omp_lock_t sum_lock, total_lock;
	omp_init_lock(&sum_lock);
	omp_init_lock(&total_lock);
	// -----------------------------------------------------------------------
	// Output a start message
	printf("Parallel Timings for %ld iterations\n\n", VERYBIG);
	// repeat experiment several times
	for (i = 0; i < 10; i++)
	{
		// get starting time56 x CHAPTER 3 PARALLEL STUDIO XE FOR THE IMPATIENT
		starttime = omp_get_wtime();
		// reset check sum & running total
		sum = 0;
		total = 0.0;
// Work Loop, do some work by looping VERYBIG times
#pragma omp parallel for private(sumx, sumy, k)
		// reduction(+: sum, total)
		// schedule(static, 2000)
		for (j = 0; j < VERYBIG; j++)
		{
			// increment check sum
			// #pragma omp critical
			omp_set_lock(&sum_lock);
			sum += 1;
			omp_unset_lock(&sum_lock);
			// Calculate first arithmetic series
			sumx = 0.0;
			for (k = 0; k < j; k++)
				sumx = sumx + (double)k;
			// Calculate second arithmetic series
			sumy = 0.0;
			for (k = j; k > 0; k--)
				sumy = sumy + (double)k;
			if (sumx > 0.0)
			{
				// #pragma omp critical
				omp_set_lock(&total_lock);
				total = total + 1.0 / sqrt(sumx);
				omp_unset_lock(&total_lock);
			}
			if (sumy > 0.0)
			{
				// #pragma omp critical
				omp_set_lock(&total_lock);
				total = total + 1.0 / sqrt(sumy);
				omp_unset_lock(&total_lock);
			}
		}
		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;
		// report elapsed time
		printf("Time Elapsed: %f Secs, Total = %lf, Check Sum = %ld\n",
			   elapsedtime, total, sum);
		times_sum += elapsedtime;
	}
	printf("\nThe average running time was: %lf\n", times_sum / 10);
	// return integer as required by function header
	getchar();
	return 0;
}
