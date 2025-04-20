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
    {177, 25, 251},  // STATE1
    {177, 25, 251},  // STATE2
    {165, 30, 233},  // STATE3
    {153, 38, 215},  // STATE4
    {139, 51, 196},  // STATE5
    {127, 64, 180},  // STATE6
    {115, 78, 162},  // STATE7
    {103, 93, 144},  // STATE8
    {91, 108, 127},  // STATE9
    {79, 124, 109},  // STATE10
    {68, 138, 92},   // STATE11
    {58, 153, 76},   // STATE12
    {50, 169, 63},   // STATE13
    {45, 185, 50},   // STATE14
    {42, 200, 42}    // STATE15
};

namespace utils {

    // TODO: Need to change bool to uint8 for using for other CA models for all of them
    void generate_random_grid(bool *grid, size_t grid_size, int seed) {
        cout << "Initializing random grid" << endl;
        srand(seed);
        int count = 0;

        for (size_t i = 0; i < grid_size; i++) {
            for (size_t j = 0; j < grid_size; j++) {
                grid[i * grid_size + j] = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
                count += grid[i * grid_size + j];
            }
        }
        cout << "Number of non zero elements: " << count << endl;
        cout << "Percent: " << (float)count / (float)(grid_size * grid_size) << endl;
    }

    void save_grid(const bool *grid, size_t grid_size) {
        string filename =
            "grid_table_" + to_string(grid_size) + "x" + to_string(grid_size) + ".bin";

        cout << "Saving table " << grid_size << "x" << grid_size << " in file " << filename << endl;

        ofstream file(filename, ios::binary);
        if (!file) {
            cerr << "Failed to open file: " << filename << endl;
            return;
        }

        file.write(reinterpret_cast<const char *>(grid), grid_size * grid_size * sizeof(bool)); // NOLINT
        file.close();
    }

    void generate_rgb(int width, int height, bool *grid, char* rgb) {
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

    void save_grid_to_png(bool *X, int gridSize, int iteration) {
        int channels = 3;
        // Create output buffer
        unique_ptr<unsigned char[]> image(new unsigned char[gridSize * gridSize * channels]);
        generate_rgb(gridSize, gridSize, X, reinterpret_cast<char*>(image.get()));

        // Ensure output directory exists
        const fs::path output_dir = "output";
        fs::create_directories(output_dir); // No-op if it already exists

        // Construct full path: output/gol_<iteration>.png
        fs::path filename = output_dir / std::format("gol_{}.png", iteration);

        stbi_write_png(filename.string().c_str(), gridSize, gridSize, channels, image.get(), gridSize * channels);
    }
} // namespace utils
