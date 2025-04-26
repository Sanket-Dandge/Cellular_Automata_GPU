#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <string>
#include <chrono>
#include <type_traits>

#define THRESHOLD 0.3
#define GENERATIONS 10
#define TOTAL_STATES 15
#define P_STATE (1.0/15.0)
#define GRID_SIZE 1024
#define ELEMENTS_PER_CELL 8
#define ROW_SIZE GRID_SIZE/ELEMENTS_PER_CELL    // Real grid dimension

namespace utils {
    void read_configuration_from_file(int *X, std::string filename, size_t N);
    void generate_random_grid(uint8_t *grid, size_t grid_size, int seed = time(nullptr),
                              uint8_t state_count = 2);
    void save_grid(uint8_t *X, size_t N);
    char* generate_rgb(int width, int height, uint64_t* grid, char* rgb);
    char* generate_rgb_packet(int width, int height, uint64_t* grid, char* rgb);
    float r4_uniform_01 (int *seed);
    void save_grid_to_png(uint8_t* X, int grid_size, int iteration);
    void save_grid_to_png_ww(uint8_t *grid, int grid_size, int iteration);
} // namespace utils

#endif // !UTILS_H
