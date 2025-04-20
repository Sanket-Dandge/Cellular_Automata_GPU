#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>

#define toLinearIndex(i, j, stride) (((i) * (stride)) + (j))

namespace kernels {
    __host__ __device__ int countNeighbors(bool *currentGrid, int col, int row, int gridSize);
    __host__ __device__ bool cyclicIsNeighbors(int *currentGrid, int col, int row, int gridSize, int index);
    __global__ void computeNextGenKernel(bool *currentGrid, bool *nextGrid, int N);
    __global__ void cyclicComputeNextGenKernel(int *currentGrid, int *nextGrid, int N);
} // namespace kernels

#endif
