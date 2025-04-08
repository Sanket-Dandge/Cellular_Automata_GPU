#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

namespace utils {
    void read_configuration_from_file(bool *X, std::string filename, size_t N) {
        FILE *file = fopen(filename.c_str(), "r+");
        if (file == NULL) {
            std::cout << "Could not open file: " << filename << std::endl;
            std::cout << "Exiting!" << std::endl;
            return;
        }

        int size = fread(X, sizeof(bool), N * N, file);
        std::cout << "elements: " << size << std::endl;
        fclose(file);
    }

    void generate_random_grid(bool *X, size_t N) {
        srand(time(NULL));
        int count = 0;

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                X[i * N + j] = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
                count += X[i * N + j];
            }
        }
        std::cout << "Number of non zero elements: " << count << std::endl;
        std::cout << "Percent: " << (float)count / (float)(N * N) << std::endl;
    }

    void save_grid(bool *X, size_t N) {
        FILE *file;
        char filename[20];
        std::cout << filename << "table " << N << "x" << N << std::endl;
        std::cout << "Saving table in file " << filename << std::endl;
        file = fopen(filename, "w+");
        fwrite(X, sizeof(int), N * N, file);
        fclose(file);
    }
} // namespace utils
