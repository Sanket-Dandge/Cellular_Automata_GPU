#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>

#define toLinearIndex(i, j, stride) (((i) * (stride)) + (j))

namespace kernels {
    __host__ __device__ int countNeighbors(bool *currentGrid, int col, int row, int gridSize);
    __global__ void simpleGhostNextGenerationKernel(bool *currentGrid, bool *nextGrid, int N);
} // namespace kernels

#endif
