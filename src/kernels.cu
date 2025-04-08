#include "kernels.cuh"
#include <cstddef>
#include <cuda.h>

namespace kernels {
    __host__ __device__ int countNeighbors(bool *currentGrid, int col, int row, int gridSize) {
        int leftCol = col - 1;
        int rightCol = col + 1;
        int rowOffset = row * gridSize;
        int topRowOffset = rowOffset - gridSize;
        int bottomRowOffset = rowOffset + gridSize;

        return currentGrid[leftCol + topRowOffset] + currentGrid[col + topRowOffset] +
               currentGrid[rightCol + topRowOffset] + currentGrid[leftCol + bottomRowOffset] +
               currentGrid[col + bottomRowOffset] + currentGrid[rightCol + bottomRowOffset] +
               currentGrid[leftCol + rowOffset] + currentGrid[rightCol + rowOffset];
    }

    __global__ void computeNextGenKernel(bool *currentGrid, bool *nextGrid, int N) {
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        if ((col < N + 1) && (row < N + 1)) {
            size_t rowOffset = row * (N + 2);
            int index = rowOffset + col;
            int livingNeighbors = kernels::countNeighbors(currentGrid, col, rowOffset, N + 2);
            nextGrid[index] =
                livingNeighbors == 3 || (livingNeighbors == 2 && currentGrid[index]) ? 1 : 0;
        }
        return;
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
