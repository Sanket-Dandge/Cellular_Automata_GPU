#include "kernels.cuh"
#include <cstddef>
#include <cuda.h>

namespace kernels {
__host__ __device__ int countNeighbors(bool *currentGrid, int centerCol, int leftCol, int rightCol,
                                       int centerRow, int topRow, int bottomRow) {
    return currentGrid[leftCol + topRow] + currentGrid[centerCol + topRow] +
           currentGrid[rightCol + topRow] + currentGrid[leftCol + bottomRow] +
           currentGrid[centerCol + bottomRow] + currentGrid[rightCol + bottomRow] +
           currentGrid[leftCol + centerRow] + currentGrid[rightCol + centerRow];
}

__global__ void simpleGhostNextGenerationKernel(bool *currentGrid, bool *nextGrid, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if ((col < N + 1) && (row < N + 1)) {
        size_t up = (row - 1) * (N + 2);
        size_t center = row * (N + 2);
        size_t down = (row + 1) * (N + 2);
        size_t left = col - 1;
        size_t right = col + 1;
        int index = center + col;
        int livingNeighbors =
            kernels::countNeighbors(currentGrid, col, left, right, center, up, down);
        nextGrid[index] =
            livingNeighbors == 3 || (livingNeighbors == 2 && currentGrid[index]) ? 1 : 0;
    }
    return;
}

__global__ void updateColorArray(GLubyte *colorArray, bool *currentGrid, bool *nextGrid, int N) {
    int indexX = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    for (int i = indexY; i < N + 1; i += strideY) {
        size_t y = __umul24(i, N + 2);
        size_t up = __umul24(i - 1, N + 2);
        size_t down = __umul24(i + 1, N + 2);
        for (int j = indexX; j < N + 1; j += strideX) {
            int index = y + j;
            int colorIndex = 3 * ((i - 1) * N + j - 1);
            if (nextGrid[index] && !currentGrid[index]) {
                colorArray[colorIndex] = 0;
                colorArray[colorIndex + 1] = 255;
            }

            if (!nextGrid[index] && currentGrid[index]) {
                colorArray[colorIndex] = 255;
                colorArray[colorIndex + 1] = 0;
            } else if (!nextGrid[index] && !currentGrid[index]) {
                colorArray[colorIndex] > 0 ? colorArray[colorIndex]-- : colorArray[colorIndex] = 0;
            }

            if (nextGrid[index]) {
                colorArray[colorIndex + 2] >= 255 ? colorArray[colorIndex + 2] = 255
                                                  : colorArray[colorIndex + 2]++;
            }
        }
    }
}

__global__ void updateGhostRows(bool *grid, int N, size_t pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (x < N - 1) {
        grid[toLinearIndex(N - 1, x, pitch)] = grid[toLinearIndex(1, x, pitch)];
        grid[toLinearIndex(0, x, pitch)] = grid[toLinearIndex(N - 2, x, pitch)];
    }
}

__global__ void updateGhostCols(bool *grid, int N, int pitch) {
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (y < N - 1) {
        grid[toLinearIndex(y, N - 1, pitch)] = grid[toLinearIndex(y, 1, pitch)];
        grid[toLinearIndex(y, 0, pitch)] = grid[toLinearIndex(y, N - 2, pitch)];
    }
}

__global__ void updateGhostCorners(bool *grid, int N, int pitch) {
    grid[toLinearIndex(0, 0, pitch)] = grid[toLinearIndex(N - 2, N - 2, pitch)];
    grid[toLinearIndex(N - 1, N - 1, pitch)] = grid[toLinearIndex(1, 1, pitch)];
    grid[toLinearIndex(0, N - 1, pitch)] = grid[toLinearIndex(N - 2, 1, pitch)];
    grid[toLinearIndex(N - 1, 0, pitch)] = grid[toLinearIndex(1, N - 2, pitch)];
}
} // namespace kernels
