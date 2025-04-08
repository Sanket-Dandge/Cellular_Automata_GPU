#include <iostream>
#include <cuda.h>
#include "kernels.cuh"

using namespace kernels;

void printGrid(bool* grid, int N) {
    for (int i = 0; i < N + 2; ++i) {
        for (int j = 0; j < N + 2; ++j) {
            std::cout << grid[i * (N + 2) + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int N = 4;  // actual grid size (ghost borders make it (N+2)x(N+2))
    const int totalSize = (N + 2) * (N + 2);

    // Allocate and initialize host memory
    bool h_current[totalSize] = {0};
    bool h_next[totalSize] = {0};

    // Make a glider pattern or any custom one you want
    h_current[1 * (N + 2) + 2] = 1;
    h_current[2 * (N + 2) + 3] = 1;
    h_current[3 * (N + 2) + 1] = 1;
    h_current[3 * (N + 2) + 2] = 1;
    h_current[3 * (N + 2) + 3] = 1;

    std::cout << "Initial Grid:\n";
    printGrid(h_current, N);

    // Allocate device memory
    bool *d_current, *d_next;
    cudaMalloc(&d_current, totalSize * sizeof(bool));
    cudaMalloc(&d_next, totalSize * sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_current, h_current, totalSize * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    simpleGhostNextGenerationKernel<<<gridSize, blockSize>>>(d_current, d_next, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_next, d_next, totalSize * sizeof(bool), cudaMemcpyDeviceToHost);

    std::cout << "Next Generation:\n";
    printGrid(h_next, N);

    // Cleanup
    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
