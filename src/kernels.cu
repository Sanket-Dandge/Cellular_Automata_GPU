#include "kernels.cuh"
#include "kernels.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                                           \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

namespace kernels {
    __host__ __device__ int countNeighbors(bool *currentGrid, int col, int row, int gridSize) {
        int leftCol = (col - 1 + gridSize) % gridSize;
        int rightCol = (col + 1) % gridSize;
        int rowOffset = row * gridSize;
        int topRowOffset = ((row - 1 + gridSize) % gridSize) * gridSize;
        int bottomRowOffset = ((row + 1) % gridSize) * gridSize;
        // int bottomRowOffset = rowOffset + gridSize;

        return currentGrid[leftCol + topRowOffset] + currentGrid[col + topRowOffset] +
               currentGrid[rightCol + topRowOffset] + currentGrid[leftCol + bottomRowOffset] +
               currentGrid[col + bottomRowOffset] + currentGrid[rightCol + bottomRowOffset] +
               currentGrid[leftCol + rowOffset] + currentGrid[rightCol + rowOffset];
    }

    __global__ void computeNextGenKernel(bool *currentGrid, bool *nextGrid, int N) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        // if (col == 0 || col == N - 1 || row == 0 || row == N - 1) {
        //     return;
        // }
        size_t rowOffset = row * N;
        int index = rowOffset + col;
        if (index >= N * N) {
            printf("%d,%d\n", col, row);
        }
        int livingNeighbors = kernels::countNeighbors(currentGrid, col, row, N);
        // int livingNeighbors = 3;
        nextGrid[index] =
            livingNeighbors == 3 || (livingNeighbors == 2 && currentGrid[index]) ? 1 : 0;
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

void computeNextGen(bool *currentGrid, bool *nextGrid, int N) {
    // Allocate device memory
    bool *d_current, *d_next;
    int totalSize = N * N;
    CUDA_CHECK(cudaMalloc(&d_current, totalSize * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next, totalSize * sizeof(bool)));

    // Copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_current, currentGrid, totalSize * sizeof(bool), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    kernels::computeNextGenKernel<<<gridSize, blockSize>>>(d_current, d_next, N);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(nextGrid, d_next, totalSize * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
}
