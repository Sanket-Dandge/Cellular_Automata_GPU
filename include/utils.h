#ifndef UTILS_H
#define UTILS_H

#include <string>

#define THRESHOLD 0.3
#define GENERATIONS 10

namespace utils {
    void read_from_file(bool *X, std::string filename, size_t N);
    void generate_table(bool *X, size_t N);
    void save_table(bool *X, size_t N);
} // namespace utils

#endif // !UTILS_H
