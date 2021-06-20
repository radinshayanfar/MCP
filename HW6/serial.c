#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define RUN_COUNT 10
double timesSum = 0;


void constantInit(int *data, int size, int val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}

int reduceSerial(int argc, char **argv, int exp) {
    int n = 1<<exp;
	size_t mem_size = sizeof(int) * n;

	int *h_A = (int *)malloc(mem_size);
    if (h_A == NULL) {
        printf("[!] A allocation failed!\n");
        exit(1);
    }

	constantInit(h_A, n, 1);

	// get starting time
	double starttime = omp_get_wtime();

    int sum = 0;

	for (int i = 0; i < n; i++) {
        sum += h_A[i];
	}

	// get ending time and use it to determine elapsed time
	double elapsedtime = (omp_get_wtime() - starttime) * 1000;

	printf("[-] Sum = %d, Elapsed time in msec = %lf\n", sum, elapsedtime);
	timesSum += elapsedtime;

	free(h_A);

    return sum;
}


/**
* Program main
*/
int main(int argc, char **argv)
{
    #ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return;
	#endif

	// Size of square matrices
	unsigned short exp = 0;
	printf("[-] Exponent = ");
	scanf("%u", &exp);
    long n = 1 << exp;

    printf("Reducing %ld size array.\n", n);

	for (int i = 0; i < RUN_COUNT; i++) {
        reduceSerial(argc, argv, exp);
	}

    double avg = timesSum / RUN_COUNT;
	printf("\n[-] Average time in msec = %lf\n", avg);
    printf("[-] Average memory bandwidth = %lf GB/s\n", n * 4.0 / (avg / 1000) / (1 << 30));
	
	return 0;
}
