#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <cuda.h>
#include <cstdio>
#define ELEMENTS_PER_CELL 8
#define ROW_SIZE GRID_SIZE/ELEMENTS_PER_CELL    // Real grid dimension

#define CUDA_CHECK(call)                                                                           \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

namespace kernels {
    __host__ __device__ int count_neighbors(bool *currentGrid, int col, int row, int gridSize);
    __host__ __device__ int cyclic_check_neighbors(uint8_t *currentGrid, int col, int row, int grid_size, int index);
    __global__ void compute_next_gen_kernel(bool *currentGrid, bool *nextGrid, int N);
    __global__ void cyclic_baseline_kernel(uint8_t *currentGrid, uint8_t *nextGrid, int N);
    __global__ void cyclic_lookup_kernel(uint8_t *currentGrid, uint8_t* nextGrid, int N, uint8_t *lookup_table);
    __global__ void cyclic_packet_coding_kernel(uint64_t *currentGrid, uint64_t* nextGrid, int N, uint8_t *lookup_table);
    __device__ void setSubCellD(uint64_t* currentCell, char position, uint8_t subCell);
    __device__ uint8_t getSubCellD(uint64_t currentCell, char position);
    namespace wireworld {
        __global__ void compute_next_gen(uint8_t *current_grid, uint8_t *next_grid, int grid_size);
    }
} // namespace kernels

#endif
