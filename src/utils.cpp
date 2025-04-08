#include "utils.h"
#include <format>
#include <memory>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

namespace utils {

    void generate_random_grid(bool *X, size_t N) {
        cout << "Initializing random grid" << endl;
        srand(time(NULL));
        int count = 0;

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                X[i * N + j] = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
                count += X[i * N + j];
            }
        }
        cout << "Number of non zero elements: " << count << endl;
        cout << "Percent: " << (float)count / (float)(N * N) << endl;
    }

    void save_grid(bool *X, size_t N) {
        FILE *file;
        char filename[20];
        cout << filename << "table " << N << "x" << N << endl;
        cout << "Saving table in file " << filename << endl;
        file = fopen(filename, "w+");
        fwrite(X, sizeof(int), N * N, file);
        fclose(file);
    }

    void save_grid_to_png(bool *X, int gridSize, int iteration) {
        // Create output buffer
        unique_ptr<unsigned char[]> image(new unsigned char[gridSize * gridSize]);
        for (int i = 0; i < gridSize * gridSize; i++) {
            image[i] = X[i] ? 255 : 0;
        }

        // Ensure output directory exists
        fs::path outputDir = "output";
        fs::create_directories(outputDir); // No-op if it already exists

        // Construct full path: output/gol_<iteration>.png
        fs::path filename = outputDir / std::format("gol_{}.png", iteration);
        // Or use: fmt::format if you don't have C++20

        stbi_write_png(filename.string().c_str(), gridSize, gridSize, 1, image.get(), gridSize);
    }
} // namespace utils
