#include "kernels.cuh"
#include "kernels.hpp"

using namespace std;

namespace kernels::gol {
    enum States : uint8_t {
        DEAD = 0,
        ALIVE = 1,
    };

    __device__ int count_neighbors(uint8_t *current_grid, uint col, uint row, int ca_grid_size) {
        uint left_col = (col - 1 + ca_grid_size) % ca_grid_size;
        uint right_col = (col + 1) % ca_grid_size;
        uint row_offset = row * ca_grid_size;
        uint top_row_offset = ((row - 1 + ca_grid_size) % ca_grid_size) * ca_grid_size;
        uint bottom_row_offset = ((row + 1) % ca_grid_size) * ca_grid_size;

        return current_grid[left_col + top_row_offset] + current_grid[col + top_row_offset] +
               current_grid[right_col + top_row_offset] +
               current_grid[left_col + bottom_row_offset] + current_grid[col + bottom_row_offset] +
               current_grid[right_col + bottom_row_offset] + current_grid[left_col + row_offset] +
               current_grid[right_col + row_offset];
    }

    __global__ void compute_next_gen_kernel(uint8_t *current_grid, uint8_t *next_grid,
                                            int ca_grid_size) {
        uint col = blockIdx.x * blockDim.x + threadIdx.x;
        uint row = blockIdx.y * blockDim.y + threadIdx.y;

        size_t row_offset = row * ca_grid_size;
        uint index = row_offset + col;
        if (index >= ca_grid_size * ca_grid_size) {
            printf("%d,%d\n", col, row);
        }
        int living_neighbors = kernels::gol::count_neighbors(current_grid, col, row, ca_grid_size);
        next_grid[index] =
            living_neighbors == 3 || (living_neighbors == 2 && current_grid[index]) ? ALIVE : DEAD;
        return;
    }

    void compute_next_gen(uint8_t *current_grid, uint8_t *next_grid, size_t ca_grid_size,
                          size_t niter) {
        // Allocate device memory
        uint8_t *d_current = nullptr, *d_next = nullptr;
        size_t total_size = ca_grid_size * ca_grid_size;
        CUDA_CHECK(cudaMalloc(&d_current, total_size * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_next, total_size * sizeof(bool)));

        // Copy data to device
        CUDA_CHECK(
            cudaMemcpy(d_current, current_grid, total_size * sizeof(bool), cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 block_size(32, 32);
        dim3 grid_size((ca_grid_size + block_size.x - 1) / block_size.x,
                       (ca_grid_size + block_size.y - 1) / block_size.y);

        int citers = 0;
        while (citers < niter) {
            kernels::gol::compute_next_gen_kernel<<<grid_size, block_size>>>(d_current, d_next,
                                                                             ca_grid_size);
            cudaDeviceSynchronize();
            niter++;
            swap(d_current, d_next);
        }

        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        // Copy result back to host
        CUDA_CHECK(
            cudaMemcpy(next_grid, d_next, total_size * sizeof(bool), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
    }

} // namespace kernels::gol
