#ifndef UTILS_H
#define UTILS_H

#include <string>

#define THRESHOLD 0.3
#define GENERATIONS 10

namespace utils {
    void read_configuration_from_file(bool *X, std::string filename, size_t N);
    void generate_random_grid(bool *X, size_t N);
    void save_grid(bool *X, size_t N);
    void save_grid_to_png(bool *X, int gridSize, int iteration);
} // namespace utils

#endif // !UTILS_H
