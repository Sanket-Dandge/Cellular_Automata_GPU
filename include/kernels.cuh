#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>

#define toLinearIndex(i, j, stride) (((i) * (stride)) + (j))

namespace kernels {
    __host__ __device__ int count_neighbors(bool *current_grid, int col, int row, int grid_size);
    __global__ void compute_next_gen_kernel(bool *current_grid, bool *next_grid, int N);
} // namespace kernels

#endif
