#include "utils.h"
#include <memory>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

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
        // Create an 8-bit grayscale buffer (1 byte per pixel)
        unique_ptr<unsigned char[]> image(new unsigned char[gridSize * gridSize]);

        for (int i = 0; i < gridSize * gridSize; i++) {
            image[i] = X[i] ? 255 : 0;
        }

        // Construct filename like "gol_42.png"
        char filename[64];
        snprintf(filename, sizeof(filename), "gol_%d.png", iteration);

        // Write PNG: width, height, channels = 1 (grayscale), stride = width * 1 byte
        stbi_write_png(filename, gridSize, gridSize, 1, image.get(), gridSize);
    }
} // namespace utils
