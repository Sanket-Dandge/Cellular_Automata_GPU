
#include "kernels.cuh"
#include <iostream>
#include <stdio.h>

#define LOOKUP_TABLE_LIMIT (17 * 8 + 1)

using namespace std;

namespace kernels::wireworld {
    enum States : uint8_t {
        EMPTY = 0,
        HEAD = 17,
        TAIL = 1,
        CONDUCTOR = 2,
    };

    void init_lookup_table(uint8_t *lookup_table_) {
        int i = 0, j = 0;
        uint8_t(*lookup_table)[LOOKUP_TABLE_LIMIT] = (uint8_t(*)[LOOKUP_TABLE_LIMIT])lookup_table_;

        // Init lookup_table:
        // 0: Empty. If a cell is EMPTY remains EMPTY
        for (j = 0; j < LOOKUP_TABLE_LIMIT; j++) {
            lookup_table[EMPTY][j] = EMPTY;
        }
        // 2: Head. A HEAD cell turns always into TAIL
        for (j = 0; j < LOOKUP_TABLE_LIMIT; j++) {
            lookup_table[HEAD][j] = TAIL;
        }
        // 3: Tail. A TAIL cell turns always into CONDUCTOR
        for (j = 0; j < LOOKUP_TABLE_LIMIT; j++) {
            lookup_table[TAIL][j] = CONDUCTOR;
        }
        // 4: Conductor. A CONDUCTOR cell turns into HEAD if exactly one or two of the neighbouring
        // cells are HEADS),
        //    otherwise remains CONDUCTOR.
        for (j = 0; j < LOOKUP_TABLE_LIMIT; j++) {
            lookup_table[CONDUCTOR][j] = CONDUCTOR;
        }
        for (j = HEAD; j < (HEAD * 2 + CONDUCTOR * 6 + 1); j++) {
            lookup_table[CONDUCTOR][j] = HEAD;
        }

        // Fill with EMPTY not used lookup_table entries
        for (i = CONDUCTOR + 1; i < HEAD - 1; i++)
            for (j = 0; j < LOOKUP_TABLE_LIMIT; j++)
                lookup_table[i][j] = EMPTY;
    }

    __device__ int count_neighbor_electron_head(uint8_t *current_grid, uint col, uint row,
                                                uint grid_size) {
        uint left_col = (col - 1 + grid_size) % grid_size;
        uint right_col = (col + 1) % grid_size;
        uint row_offset = row * grid_size;
        uint top_row_offset = ((row - 1 + grid_size) % grid_size) * grid_size;
        uint bottom_row_offset = ((row + 1) % grid_size) * grid_size;

        return (current_grid[left_col + top_row_offset] == States::HEAD) +
               (current_grid[col + top_row_offset] == States::HEAD) +
               (current_grid[right_col + top_row_offset] == States::HEAD) +
               (current_grid[left_col + bottom_row_offset] == States::HEAD) +
               (current_grid[col + bottom_row_offset] == States::HEAD) +
               (current_grid[right_col + bottom_row_offset] == States::HEAD) +
               (current_grid[left_col + row_offset] == States::HEAD) +
               (current_grid[right_col + row_offset] == States::HEAD);
    }

    __device__ int surround_sum(uint8_t *grd, uint col, uint row, uint grid_size) {
        uint left = (col - 1 + grid_size) % grid_size;
        uint right = (col + 1) % grid_size;
        uint row_offset = row * grid_size;
        uint top_offset = ((row - 1 + grid_size) % grid_size) * grid_size;
        uint bottom_offset = ((row + 1) % grid_size) * grid_size;

        // clang-format off
        return grd[left + top_offset]    + grd[col + top_offset]    + grd[right + top_offset] +
               grd[left + row_offset]                               + grd[right + row_offset] +
               grd[left + bottom_offset] + grd[col + bottom_offset] + grd[right + bottom_offset];
        // clang-format on
    }

    __global__ void compute_next_gen_kernel_lut(uint8_t *current_grid, uint8_t *next_grid,
                                                uint grid_size, uint8_t *lut) {
        uint col = blockIdx.x * blockDim.x + threadIdx.x;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t row_offset = row * grid_size;
        size_t index = row_offset + col;

        if (index >= grid_size * grid_size) {
            printf("%d,%d\n", col, row);
        }
        uint8_t(*lookup_table)[LOOKUP_TABLE_LIMIT] = (uint8_t(*)[LOOKUP_TABLE_LIMIT])lut;

        int sum = surround_sum(current_grid, col, row, grid_size);
        next_grid[index] = lookup_table[current_grid[index]][sum];
    }
    __global__ void compute_next_gen_kernel_base(uint8_t *current_grid, uint8_t *next_grid,
                                                 uint grid_size) {
        uint col = blockIdx.x * blockDim.x + threadIdx.x;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t row_offset = row * grid_size;
        size_t index = row_offset + col;
        if (index >= grid_size * grid_size) {
            printf("%d,%d\n", col, row);
        }
        if (current_grid[index] == HEAD) {
            next_grid[index] = TAIL;
        } else if (current_grid[index] == TAIL) {
            next_grid[index] = CONDUCTOR;
        } else if (current_grid[index] == CONDUCTOR) {
            int cnt = count_neighbor_electron_head(current_grid, col, row, grid_size);
            next_grid[index] = cnt == 1 || cnt == 2 ? HEAD : CONDUCTOR;
        }
    }

    __global__ void compute_next_gen_kernel_packet_coding(uint8_t *current_grid, uint8_t *next_grid,
                                                          uint grid_size, uint8_t *lut) {
        uint col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t row_offset = row * grid_size;
        size_t index = row_offset + col;

        if (index >= grid_size * grid_size) {
            printf("%d,%d\n", col, row);
        }
        uint8_t(*lookup_table)[LOOKUP_TABLE_LIMIT] = (uint8_t(*)[LOOKUP_TABLE_LIMIT])lut;

        for(uint i = 0; i < 8; i++){
            int sum = surround_sum(current_grid, col + i, row, grid_size);
            next_grid[index + i] = lookup_table[current_grid[index + i]][sum];
        }
    }

    void compute_next_gen_base(uint8_t *current_grid, uint ca_grid_size, size_t niter) {
        // Allocate device memory
        uint8_t *d_current = nullptr, *d_next = nullptr;
        size_t total_size = ca_grid_size * ca_grid_size;
        CUDA_CHECK(cudaMalloc(&d_current, total_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_next, total_size * sizeof(uint8_t)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_current, current_grid, total_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 block_size(32, 32);
        dim3 grid_size((ca_grid_size + block_size.x - 1) / block_size.x,
                       (ca_grid_size + block_size.y - 1) / block_size.y);

        int citers = 0;
        while (citers < niter) {
            kernels::wireworld::compute_next_gen_kernel_base<<<grid_size, block_size>>>(
                d_current, d_next, ca_grid_size);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            citers++;
            swap(d_current, d_next);
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(current_grid, d_current, total_size * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
    }
    void compute_next_gen_lut(uint8_t *current_grid, uint ca_grid_size, size_t niter) {
        auto lut_size = sizeof(uint8_t) * (HEAD + 1) * LOOKUP_TABLE_LIMIT;
        auto *lut = (uint8_t *)malloc(lut_size); // NOLINT
        init_lookup_table(lut);

        // Allocate device memory
        uint8_t *d_current = nullptr, *d_next = nullptr, *d_lut = nullptr;
        size_t total_size = ca_grid_size * ca_grid_size;
        CUDA_CHECK(cudaMalloc(&d_current, total_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_next, total_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_lut, lut_size));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_current, current_grid, total_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lut, lut, lut_size, cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 block_size(32, 32);
        dim3 grid_size((ca_grid_size + block_size.x - 1) / block_size.x,
                       (ca_grid_size + block_size.y - 1) / block_size.y);

        int citers = 0;
        while (citers < niter) {
            kernels::wireworld::compute_next_gen_kernel_lut<<<grid_size, block_size>>>(
                d_current, d_next, ca_grid_size, d_lut);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            citers++;
            swap(d_current, d_next);
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(current_grid, d_current, total_size * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
        free(lut); // NOLINT
    }

    void compute_next_gen_packet_coding(uint8_t *current_grid, uint ca_grid_size, size_t niter) {
        if (ca_grid_size < 256) {
            cerr << "Grid too small: " << ca_grid_size << endl;
        }
        auto lut_size = sizeof(uint8_t) * (HEAD + 1) * LOOKUP_TABLE_LIMIT;
        auto *lut = (uint8_t *)malloc(lut_size); // NOLINT
        init_lookup_table(lut);

        // Allocate device memory
        uint8_t *d_current = nullptr, *d_next = nullptr, *d_lut = nullptr;
        size_t total_size = ca_grid_size * ca_grid_size;
        CUDA_CHECK(cudaMalloc(&d_current, total_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_next, total_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_lut, lut_size));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_current, current_grid, total_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lut, lut, lut_size, cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 block_size(32, 32);
        dim3 grid_size((ca_grid_size + block_size.x - 1) / (block_size.x * 8),
                       (ca_grid_size + block_size.y - 1) / (block_size.y));

        int citers = 0;
        while (citers < niter) {
            kernels::wireworld::compute_next_gen_kernel_packet_coding<<<grid_size, block_size>>>(
                d_current, d_next, ca_grid_size, d_lut);
            CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
            citers++;
            swap(d_current, d_next);
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(current_grid, d_current, total_size * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
        free(lut); // NOLINT
    }
} // namespace kernels::wireworld
