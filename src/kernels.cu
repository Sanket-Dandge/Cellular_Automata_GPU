#include "cyclic_ca.hpp"
#include "kernels.cuh"
#include "kernels.hpp"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda.h>

#define ROW_SIZE GRID_SIZE / ELEMENTS_PER_CELL // Real grid dimension

namespace kernels {
    __host__ __device__ int cyclic_check_neighbors(uint8_t *currentGrid, int col, int row,
                                                   int grid_size, int index) {
        int leftCol = (col - 1 + grid_size) % grid_size;
        int rightCol = (col + 1) % grid_size;
        int rowOffset = row * grid_size;
        int topRowOffset = ((row - 1 + grid_size) % grid_size) * grid_size;
        int bottomRowOffset = ((row + 1) % grid_size) * grid_size;
        int nextState = (currentGrid[index] + 1) % TOTAL_STATES;

        return ((currentGrid[col + topRowOffset] == nextState) ||
                (currentGrid[col + bottomRowOffset] == nextState) ||
                (currentGrid[rowOffset + leftCol] == nextState) ||
                (currentGrid[rowOffset + rightCol] == nextState));
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

    __global__ void cyclic_lookup_kernel(uint8_t *currentGrid, uint8_t *nextGrid, int N,
                                         uint8_t *lookup_table) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t rowOffset = row * N;
        int index = rowOffset + col;
        int current_cell = currentGrid[index];
        if (index >= N * N) {
            printf("%d,%d\n", col, row);
        }
        uint8_t nextStateNeighbor =
            kernels::cyclic_check_neighbors(currentGrid, col, row, N, index);
        // printf("(%d, %d, %d)\n", row, col, lookup_table[current_cell * 2 + nextStateNeighbor]);
        nextGrid[index] = lookup_table[current_cell * 2 + nextStateNeighbor];
    }

    __device__ uint8_t getSubCellD(uint64_t currentCell, char position) {
        return (currentCell >> ((ELEMENTS_PER_CELL - 1 - position) * 8)) & 0xFF;
    }

    __device__ void setSubCellD(uint64_t *currentCell, char position, uint8_t subCell) {
        uint64_t mask = 0xFF;
        uint64_t newCellMask = subCell;

        // Erase pos content in cell:
        mask = mask << (ELEMENTS_PER_CELL - 1 - position) * 8;
        mask = ~mask;
        *currentCell = *currentCell & mask;

        // Add subcell content to cell in pos:
        *currentCell = *currentCell | (newCellMask << (ELEMENTS_PER_CELL - 1 - position) * 8);
    }

    __global__ void cyclic_packet_coding_kernel(uint64_t *currentGrid, uint64_t *nextGrid, int N,
                                                uint8_t *lookup_table) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < GRID_SIZE && col < ROW_SIZE) {
            int leftCol = (col - 1 + (ROW_SIZE)) % ROW_SIZE;
            int rightCol = (col + 1) % ROW_SIZE;
            int rowOffset = row * ROW_SIZE;
            int topRowOffset = ((row - 1 + GRID_SIZE) % GRID_SIZE) * ROW_SIZE;
            int bottomRowOffset = ((row + 1) % GRID_SIZE) * ROW_SIZE;

            int index = row * ROW_SIZE + col;
            uint64_t currentCell = currentGrid[index];
            uint64_t nextCell = 0;

            // First subcell
            uint8_t subcell = getSubCellD(currentCell, 0);
            uint64_t upCell = currentGrid[col + topRowOffset];
            uint64_t downCell = currentGrid[col + bottomRowOffset];
            uint64_t leftCell = currentGrid[rowOffset + leftCol];
            uint64_t rightCell = currentGrid[rightCol + rowOffset];

            int nextStateNeighbor =
                (getSubCellD(upCell, 0) == (subcell + 1) % TOTAL_STATES) ||
                (getSubCellD(downCell, 0) == (subcell + 1) % TOTAL_STATES) ||
                (getSubCellD(leftCell, ELEMENTS_PER_CELL - 1) == (subcell + 1) % TOTAL_STATES) ||
                (getSubCellD(currentCell, 1) == (subcell + 1) % TOTAL_STATES);

            setSubCellD(&nextCell, 0, lookup_table[subcell * 2 + nextStateNeighbor]);

            for (int k = 1; k < ELEMENTS_PER_CELL - 1; k++) {
                subcell = getSubCellD(currentCell, k);
                nextStateNeighbor =
                    (getSubCellD(upCell, k) == (subcell + 1) % TOTAL_STATES) ||
                    (getSubCellD(downCell, k) == (subcell + 1) % TOTAL_STATES) ||
                    (getSubCellD(currentCell, k - 1) == (subcell + 1) % TOTAL_STATES) ||
                    (getSubCellD(currentCell, k + 1) == (subcell + 1) % TOTAL_STATES);

                setSubCellD(&nextCell, k, lookup_table[subcell * 2 + nextStateNeighbor]);
            }

            subcell = getSubCellD(currentCell, ELEMENTS_PER_CELL - 1);
            nextStateNeighbor =
                (getSubCellD(upCell, ELEMENTS_PER_CELL - 1) == (subcell + 1) % TOTAL_STATES) ||
                (getSubCellD(downCell, ELEMENTS_PER_CELL - 1) == (subcell + 1) % TOTAL_STATES) ||
                (getSubCellD(currentCell, ELEMENTS_PER_CELL - 2) == (subcell + 1) % TOTAL_STATES) ||
                (getSubCellD(rightCell, 0) == (subcell + 1) % TOTAL_STATES);

            setSubCellD(&nextCell, ELEMENTS_PER_CELL - 1,
                        lookup_table[subcell * 2 + nextStateNeighbor]);
            nextGrid[index] = nextCell;
        }
    }
} // namespace kernels

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

    CUDA_CHECK(cudaMemcpy(d_lookup_table, lookup_table, TOTAL_STATES * sizeof(uint8_t) * 2,
                          cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    kernels::cyclic_lookup_kernel<<<gridSize, blockSize>>>(d_current, d_next, N, d_lookup_table);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(nextGrid, d_next, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_lookup_table));
}

uint8_t getSubCellH(uint64_t currentCell, char position) {
    return (currentCell >> (ELEMENTS_PER_CELL - 1 - position) * 8);
}

void setSubCellH(uint64_t *currentCell, char position, uint8_t subCell) {
    uint64_t mask = 0xFF;
    uint64_t newCellMask = subCell;
    mask = mask << (ELEMENTS_PER_CELL - 1 - position) * 8;
    mask = ~mask;
    *currentCell = *currentCell & mask;
    *currentCell = *currentCell | (newCellMask << (ELEMENTS_PER_CELL - 1 - position) * 8);
}

void cyclic_packet_coding_gen(uint64_t *currentGrid, uint64_t *nextGrid, int N) {
    uint64_t *d_current, *d_next;
    uint8_t lookup_table[TOTAL_STATES][2], *d_lookup_table;
    int totalSize = N * N;
    CUDA_CHECK(cudaMalloc(&d_current, totalSize * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_next, totalSize * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_lookup_table, TOTAL_STATES * sizeof(uint8_t) * 2));

    CyclicCA::create_lookup_table(lookup_table);

    // Copy data to device
    CUDA_CHECK(
        cudaMemcpy(d_current, currentGrid, totalSize * sizeof(uint64_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_lookup_table, lookup_table, TOTAL_STATES * sizeof(uint8_t) * 2,
                          cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    kernels::cyclic_packet_coding_kernel<<<gridSize, blockSize>>>(d_current, d_next, N,
                                                                  d_lookup_table);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(nextGrid, d_next, totalSize * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_lookup_table));
}
