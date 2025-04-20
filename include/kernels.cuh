#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>

namespace kernels {
    __host__ __device__ int count_neighbors(bool *currentGrid, int col, int row, int gridSize);
    __host__ __device__ bool cyclic_check_neighbors(int *currentGrid, int col, int row, int gridSize, int index);
    __global__ void compute_next_gen_kernel(bool *currentGrid, bool *nextGrid, int N);
    __global__ void cyclic_compute_next_gen_kernel(int *currentGrid, int *nextGrid, int N);
} // namespace kernels

#endif
