#include <cuda_runtime.h>
#include <stdio.h>

__global__ void printWhoAmIKernel() {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    printf("Hello CUDA! I'm thread %d from block %d.\n", tx, bx);
}

int main(int argc, char **argv) {
    dim3 BLOCK_SIZE(8, 1, 1);
    dim3 GRID_SIZE(4, 1, 1);

    printWhoAmIKernel<<<GRID_SIZE, BLOCK_SIZE>>>();

    return 0;
}
