#include "kernels.cuh"
#include "kernels.hpp"
#include "cyclic_ca.hpp"
#include "utils.h"
#include <cstddef>
#include <cstdint>
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

    __host__ __device__ int cyclic_check_neighbors(uint8_t *currentGrid, int col, int row, int grid_size, int index) {
        int leftCol = (col - 1 + grid_size) % grid_size;
        int rightCol = (col + 1) % grid_size;
        int rowOffset = row * grid_size;
        int topRowOffset = ((row - 1 + grid_size) % grid_size) * grid_size;
        int bottomRowOffset = ((row + 1) % grid_size) * grid_size;
        int nextState = (currentGrid[index] + 1) % TOTAL_STATES;

        return (
            ( currentGrid[col + topRowOffset] == nextState )
            || ( currentGrid[col + bottomRowOffset] == nextState )
            || ( currentGrid[rowOffset + leftCol] == nextState )
            || ( currentGrid[rowOffset + rightCol] == nextState )
        );
    }


    __global__ void cyclic_baseline_kernel(uint8_t *currentGrid, uint8_t *nextGrid, int N) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t rowOffset = row * N;
        int index = rowOffset + col;
        int current_cell = currentGrid[index];
        if (index >= N * N) {
            printf("%d,%d\n", col, row);
        }
        int nextStateNeighbor = kernels::cyclic_check_neighbors(currentGrid, col, row, N, index);
        if (current_cell == STATE1 && nextStateNeighbor) {
            nextGrid[index] = STATE2;
        } else if (current_cell == STATE2 && nextStateNeighbor) {
            nextGrid[index] = STATE3;
        } else if (current_cell == STATE3 && nextStateNeighbor) {
            nextGrid[index] = STATE4;
        } else if (current_cell == STATE4 && nextStateNeighbor) {
            nextGrid[index] = STATE5;
        } else if (current_cell == STATE5 && nextStateNeighbor) {
            nextGrid[index] = STATE6;
        } else if (current_cell == STATE6 && nextStateNeighbor) {
            nextGrid[index] = STATE7;
        } else if (current_cell == STATE7 && nextStateNeighbor) {
            nextGrid[index] = STATE8;
        } else if (current_cell == STATE8 && nextStateNeighbor) {
            nextGrid[index] = STATE9;
        } else if (current_cell == STATE9 && nextStateNeighbor) {
            nextGrid[index] = STATE10;
        } else if (current_cell == STATE10 && nextStateNeighbor) {
            nextGrid[index] = STATE11;
        } else if (current_cell == STATE11 && nextStateNeighbor) {
            nextGrid[index] = STATE12;
        } else if (current_cell == STATE12 && nextStateNeighbor) {
            nextGrid[index] = STATE13;
        } else if (current_cell == STATE13 && nextStateNeighbor) {
            nextGrid[index] = STATE14;
        } else if (current_cell == STATE14 && nextStateNeighbor) {
            nextGrid[index] = STATE15;
        } else if (current_cell == STATE15 && nextStateNeighbor) {
            nextGrid[index] = STATE1;
        } else {
            nextGrid[index] = current_cell;
        }
        return;
    }

    // TODO: Complete It
    __global__ void cyclic_lookup_kernel(uint8_t *currentGrid, uint8_t* nextGrid, int N, uint8_t* lookup_table) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t rowOffset = row * N;
        int index = rowOffset + col;
        int current_cell = currentGrid[index];
        if (index >= N * N) {
            printf("%d,%d\n", col, row);
        }
        uint8_t nextStateNeighbor = kernels::cyclic_check_neighbors(currentGrid, col, row, N, index);
        // printf("(%d, %d, %d)\n", row, col, lookup_table[current_cell * 2 + nextStateNeighbor]);
        nextGrid[index] = lookup_table[current_cell * 2 + nextStateNeighbor];
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

void cyclic_baseline(uint8_t *currentGrid, uint8_t *nextGrid, int N) {
    // Allocate device memory
    uint8_t *d_current, *d_next;
    int totalSize = N * N;
    CUDA_CHECK(cudaMalloc(&d_current, totalSize * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_next, totalSize * sizeof(uint8_t)));

    // Copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_current, currentGrid, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    kernels::cyclic_baseline_kernel<<<gridSize, blockSize>>>(d_current, d_next, N);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(nextGrid, d_next, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
}

void cyclic_lookup_gen(uint8_t *currentGrid, uint8_t *nextGrid, int N) {
    uint8_t *d_current, *d_next, lookup_table[TOTAL_STATES][2], *d_lookup_table;
    int totalSize = N * N;
    CUDA_CHECK(cudaMalloc(&d_current, totalSize * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_next, totalSize * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_lookup_table, TOTAL_STATES * sizeof(uint8_t) * 2));

    CyclicCA::create_lookup_table(lookup_table);

    // Copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_current, currentGrid, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_lookup_table, lookup_table, TOTAL_STATES * sizeof(uint8_t) * 2, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    // int linGrid = (int)ceil(GRID_SIZE/(float)32);
    // dim3 gridSize(linGrid,linGrid);
    kernels::cyclic_lookup_kernel<<<gridSize, blockSize>>>(d_current, d_next, N, d_lookup_table);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(nextGrid, d_next, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_lookup_table));
}
