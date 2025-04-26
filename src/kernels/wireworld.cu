
#include "kernels.cuh"


namespace kernels::wireworld {
    enum States : uint8_t {
        EMPTY = 0,
        HEAD = 1,
        TAIL = 2,
        CONDUCTOR = 3,
    };

    __host__ __device__ int count_neighbor_electron_head(uint8_t *current_grid, uint col, uint row,
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

    __global__ void compute_next_gen_kernel(uint8_t *current_grid, uint8_t *next_grid,
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

    void compute_next_gen(uint8_t *current_grid, uint8_t *next_grid, uint ca_grid_size) {
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
        kernels::wireworld::compute_next_gen_kernel<<<grid_size, block_size>>>(d_current, d_next,
                                                                               ca_grid_size);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        // Copy result back to host
        CUDA_CHECK(
            cudaMemcpy(next_grid, d_next, total_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
    }
} // namespace kernels::wireworld
