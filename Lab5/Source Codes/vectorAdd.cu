/*
*	In His Exalted Name
*	Vector Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	21/05/2018
*/

#define RUN_COUNT 10

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int *allocateVector(int size);
void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);

int main()
{
	#ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

	const int vectorSize = 1024;
	int *a, *b, *c;

	double starttime, elapsedtime;
	double times_sum = 0;

	for (int i = 0; i < RUN_COUNT; i++)
	{
		// get starting time
		starttime = get_wall_time();

		a = allocateVector(vectorSize);
		b = allocateVector(vectorSize);
		c = allocateVector(vectorSize);
		fillVector(a, vectorSize);
		fillVector(b, vectorSize);
		
		addVector(a, b, c, vectorSize);

		// printVector(c, vectorSize);

		// get ending time and use it to determine elapsed time
		elapsedtime = get_wall_time() - starttime;

		// report elapsed time
		printf("[-] Time Elapsed: %f Secs\n", elapsedtime);
		times_sum += elapsedtime;
	}

	printf("\n[-] The average running time was: %lf\n", times_sum / RUN_COUNT);	

	return EXIT_SUCCESS;
}

// Allocates vector in host 
int *allocateVector(int size) {
	return (int *) malloc(sizeof(int) * size);
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}
