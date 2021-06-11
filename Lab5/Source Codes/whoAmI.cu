#include <cuda_runtime.h>
#include <stdio.h>

__global__ void whoAmIKernel(int *block, int *warp, int *local_index) {
    int bd = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int global_index = bd * bx + tx;
    block[global_index]         = bx;
    warp[global_index]          = tx / warpSize;
    local_index[global_index]   = tx;
}

int main(int argc, char **argv) {
    dim3 NUM_THREADS(64, 1, 1);
    dim3 NUM_BLOCKS(2, 1, 1);

    int size = NUM_THREADS.x * NUM_BLOCKS.x;

    int *block, *warp, *local_index;
    cudaMallocManaged(&block, size * sizeof(int));
    cudaMallocManaged(&warp, size * sizeof(int));
    cudaMallocManaged(&local_index, size * sizeof(int));

    whoAmIKernel<<<NUM_BLOCKS, NUM_THREADS>>>(block, warp, local_index);
    cudaDeviceSynchronize();

    for (int i = 0; i < size; i++) {
        printf("Calculated Thread: %d,\tBlock: %d,\tWarp %d,\tThread %d\n", i, block[i], warp[i], local_index[i]);
    }

    return 0;
}
