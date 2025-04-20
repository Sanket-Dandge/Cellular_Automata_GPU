#include "kernels.cuh"
#include "kernels.hpp"
#include <cstddef>
#include <cstdio>
#include <cuda.h>

#define CUDA_CHECK(call)                                                                           \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

namespace kernels {
    __host__ __device__ int count_neighbors(bool *current_grid, int col, int row, int grid_size) {
        int left_col = (col - 1 + grid_size) % grid_size;
        int right_col = (col + 1) % grid_size;
        int row_offset = row * grid_size;
        int top_row_offset = ((row - 1 + grid_size) % grid_size) * grid_size;
        int bottom_row_offset = ((row + 1) % grid_size) * grid_size;

        return current_grid[left_col + top_row_offset] + current_grid[col + top_row_offset] +
               current_grid[right_col + top_row_offset] + current_grid[left_col + bottom_row_offset] +
               current_grid[col + bottom_row_offset] + current_grid[right_col + bottom_row_offset] +
               current_grid[left_col + row_offset] + current_grid[right_col + row_offset];
    }

    __global__ void compute_next_gen_kernel(bool *current_grid, bool *next_grid, int N) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t row_offset = row * N;
        int index = row_offset + col;
        if (index >= N * N) {
            printf("%d,%d\n", col, row);
        }
        int living_neighbors = kernels::count_neighbors(current_grid, col, row, N);
        next_grid[index] =
            living_neighbors == 3 || (living_neighbors == 2 && current_grid[index]) ? true : false;
        return;
    }

    __host__ __device__ bool cyclic_check_neighbors(int *currentGrid, int col, int row, int gridSize, int index) {
        int leftCol = (col - 1 + gridSize) % gridSize;
        int rightCol = (col + 1) % gridSize;
        int rowOffset = row * gridSize;
        int topRowOffset = ((row - 1 + gridSize) % gridSize) * gridSize;
        int bottomRowOffset = ((row + 1) % gridSize) * gridSize;
        int nextState = (currentGrid[index] + 1) % 15;

        return (
            ( currentGrid[col + topRowOffset] == nextState )
            || ( currentGrid[col + bottomRowOffset] == nextState )
            || ( currentGrid[rowOffset + leftCol] == nextState )
            || ( currentGrid[rowOffset + rightCol] == nextState )
        );
    }


    __global__ void cyclic_compute_next_gen_kernel(int *currentGrid, int *nextGrid, int N) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t rowOffset = row * N;
        int index = rowOffset + col;
        if (index >= N * N) {
            printf("%d,%d\n", col, row);
        }
        bool nextStateNeighbor = kernels::cyclic_check_neighbors(currentGrid, col, row, N, index);
        nextGrid[index] = nextStateNeighbor ? ((currentGrid[index] + 1) % 15) : currentGrid[index];
        return;
    }
} // namespace kernels

void compute_next_gen(bool *current_grid, bool *next_grid, size_t ca_grid_size) {
    // Allocate device memory
    bool *d_current = nullptr, *d_next = nullptr;
    size_t total_size = ca_grid_size * ca_grid_size;
    CUDA_CHECK(cudaMalloc(&d_current, total_size * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next, total_size * sizeof(bool)));

    // Copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_current, current_grid, total_size * sizeof(bool), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block_size(32, 32);
    dim3 grid_size((ca_grid_size + block_size.x - 1) / block_size.x, (ca_grid_size + block_size.y - 1) / block_size.y);
    kernels::compute_next_gen_kernel<<<grid_size, block_size>>>(d_current, d_next, ca_grid_size);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(next_grid, d_next, total_size * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
}

void cyclic_compute_next_gen(int *currentGrid, int *nextGrid, int N) {
    // Allocate device memory
    int *d_current, *d_next;
    int totalSize = N * N;
    CUDA_CHECK(cudaMalloc(&d_current, totalSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next, totalSize * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_current, currentGrid, totalSize * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    kernels::cyclic_compute_next_gen_kernel<<<gridSize, blockSize>>>(d_current, d_next, N);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(nextGrid, d_next, totalSize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
}
