#include "utils.h"
#include <cstdint>
#include <format>
#include <memory>
#include <sys/types.h>
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

static const unsigned char STATE_COLORS[15][3] = {
    {255, 0, 0},     // STATE1 - Red
    {0, 255, 0},     // STATE2 - Green
    {0, 0, 255},     // STATE3 - Blue
    {255, 255, 0},   // STATE4 - Yellow
    {255, 0, 255},   // STATE5 - Magenta
    {0, 255, 255},   // STATE6 - Cyan
    {128, 0, 0},     // STATE7 - Maroon
    {0, 128, 0},     // STATE8 - Dark Green
    {0, 0, 128},     // STATE9 - Navy
    {128, 128, 0},   // STATE10 - Olive
    {128, 0, 128},   // STATE11 - Purple
    {0, 128, 128},   // STATE12 - Teal
    {192, 192, 192}, // STATE13 - Silver
    {255, 165, 0},   // STATE14 - Orange
    {0, 0, 0}        // STATE15 - Black
};

namespace utils {

    void generate_random_grid(int *X, size_t N, int seed) {
        cout << "Initializing random grid" << endl;
        srand(seed);
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

    void save_grid(int *X, size_t N) {
        FILE *file;
        char filename[20];
        cout << filename << "table " << N << "x" << N << endl;
        cout << "Saving table in file " << filename << endl;
        file = fopen(filename, "w+");
        fwrite(X, sizeof(int), N * N, file);
        fclose(file);
    }

    void generate_rgb(int width, int height, int *grid, char* rgb) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                int color_index = grid[index] - 1;
                rgb[3 * index + 0] = STATE_COLORS[color_index][0];
                rgb[3 * index + 1] = STATE_COLORS[color_index][1];
                rgb[3 * index + 2] = STATE_COLORS[color_index][2];
            }
        }
    }

    void save_grid_to_png(int *X, int gridSize, int iteration) {
        int channels = 3;
        // Create output buffer
        unique_ptr<unsigned char[]> image(new unsigned char[gridSize * gridSize * channels]);
        // for (int i = 0; i < gridSize * gridSize; i++) {
        //     image[i] = X[i] ? 255 : 0;
        // }
        generate_rgb(gridSize, gridSize, X, reinterpret_cast<char*>(image.get()));

        // Ensure output directory exists
        fs::path outputDir = "output";
        fs::create_directories(outputDir); // No-op if it already exists

        // Construct full path: output/gol_<iteration>.png
        fs::path filename = outputDir / std::format("gol_{}.png", iteration);
        // Or use: fmt::format if you don't have C++20

        stbi_write_png(filename.string().c_str(), gridSize, gridSize, channels, image.get(), gridSize * channels);
    }
} // namespace utils
