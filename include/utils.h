#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <string>
#include <chrono>

#define THRESHOLD 0.3
#define GENERATIONS 10
#define TOTAL_STATES 15
#define P_STATE (1.0/15.0)

namespace utils {
    void read_configuration_from_file(int *X, std::string filename, size_t N);
    void generate_random_grid(uint8_t *grid, size_t grid_size, int seed = time(nullptr));
    void save_grid(uint8_t *X, size_t N);
    void save_grid_to_png(uint8_t *X, int grid_size, int iteration);
    char* generate_rgb(int width, int height, uint8_t* grid, char* rgb);
    float r4_uniform_01 (int *seed);
} // namespace utils

#endif // !UTILS_H
