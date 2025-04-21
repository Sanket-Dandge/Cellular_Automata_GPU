#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <cuda.h>

namespace kernels {
    __host__ __device__ int count_neighbors(bool *currentGrid, int col, int row, int gridSize);
    __host__ __device__ bool cyclic_check_neighbors(uint8_t *currentGrid, int col, int row, int grid_size, int index);
    __global__ void compute_next_gen_kernel(bool *currentGrid, bool *nextGrid, int N);
    __global__ void cyclic_compute_next_gen_kernel(uint8_t *currentGrid, uint8_t *nextGrid, int N);
} // namespace kernels

#endif
