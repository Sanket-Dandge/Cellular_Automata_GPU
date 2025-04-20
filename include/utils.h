#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <chrono>

#define THRESHOLD 0.3
#define GENERATIONS 10

namespace utils {
    void read_configuration_from_file(int *X, std::string filename, size_t N);
    void generate_random_grid(int *X, size_t N, int seed = time(NULL));
    void save_grid(int *X, size_t N);
    void save_grid_to_png(int *X, int gridSize, int iteration);
    void generate_rgb(int width, int height, int* grid, char* rgb);
} // namespace utils

#endif // !UTILS_H
