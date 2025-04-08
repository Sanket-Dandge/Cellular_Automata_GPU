#ifndef UTILS_H
#define UTILS_H

#include <string>

#define THRESHOLD 0.3
#define GENERATIONS 10

namespace utils {
    void read_configuration_from_file(bool *grid, std::string filename, size_t grid_size);
    void generate_random_grid(bool *grid, size_t grid_size);
    void save_grid(bool *grid, size_t grid_size);
    void save_grid_to_png(const bool *grid, size_t grid_size, int iteration);
} // namespace utils

#endif // !UTILS_H
