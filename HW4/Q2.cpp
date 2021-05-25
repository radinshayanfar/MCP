#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <omp.h>

#define VERYBIG 1000

int main(int argc, char const *argv[])
{
    #ifndef _OPENMP
		printf("OpenMP is not supported.\n");
		return 0;
	#endif

    printf("Starting...\n");

    omp_lock_t lock1, lock2;
    omp_init_lock(&lock1);
    omp_init_lock(&lock2);

    for (int i = 0; i < 4; i++)
    {
        printf("i = %d\n", i);
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                {
                    omp_set_lock(&lock1);
                    for (int i = 0; i < VERYBIG; i++);

                    omp_set_lock(&lock2);
                    for (int i = 0; i < VERYBIG; i++);

                    omp_unset_lock(&lock1);
                    omp_unset_lock(&lock2);
                }
                #pragma omp section
                {
                    omp_set_lock(&lock2);
                    for (int i = 0; i < VERYBIG; i++);

                    omp_set_lock(&lock1);
                    for (int i = 0; i < VERYBIG; i++);

                    omp_unset_lock(&lock1);
                    omp_unset_lock(&lock2);
                }
            }
        }
    }

    printf("Exiting...\n");

    return 0;
}
