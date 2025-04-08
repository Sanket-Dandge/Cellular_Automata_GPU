#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <GL/gl.h>

namespace kernels {
  __host__ __device__ int countNeighbors(bool* currentGrid, int centerCol, int leftCol, int rightCol, int centerRow, int topRow, int bottomRow);
  __global__ void simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N);
}

#endif
