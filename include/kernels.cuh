#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#define ELEMENTS_PER_CELL 8
#define ROW_SIZE GRID_SIZE / ELEMENTS_PER_CELL // Real grid dimension

#define CUDA_CHECK(call)                                                                           \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#endif
