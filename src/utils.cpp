#include "utils.h"
#include "kernels.hpp"
#include <complex>
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
#include <type_traits>

using namespace std;
namespace fs = std::filesystem;


const unsigned char STATE_COLORS[15][3] = {
    {177,  25, 251},  // STATE1
    {177,  25, 251},  // STATE2
    {165,  30, 233},  // STATE3
    {153,  38, 215},  // STATE4
    {139,  51, 196},  // STATE5
    {127,  64, 180},  // STATE6
    {115,  78, 162},  // STATE7
    {103,  93, 144},  // STATE8
    { 91, 108, 127},  // STATE9
    { 79, 124, 109},  // STATE10
    { 68, 138,  92},  // STATE11
    { 58, 153,  76},  // STATE12
    { 50, 169,  63},  // STATE13
    { 45, 185,  50},  // STATE14
    { 42, 200,  42}   // STATE15
};

namespace utils {

    void generate_random_grid(uint8_t *grid, size_t grid_size, int seed) {
        cout << "Initializing random grid" << endl;
        srand(seed);
        int count = 0;

        for (size_t i = 0; i < grid_size; i++) {
            for (size_t j = 0; j < grid_size; j++) {
                bool normalize = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
                grid[i * grid_size + j] = static_cast<uint8_t>(normalize);
                count += normalize;
            }
        }
        cout << "Number of non zero elements: " << count << endl;
        cout << "Percent: " << static_cast<float>(count) / (float)(grid_size * grid_size) << endl;
    }

    float r4_uniform_01(int *seed) {
        const int i4_huge = 2147483647;
        int k;
        float r;
        
        k = *seed / 127773;
        *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
        if ( *seed < 0 ) {
            *seed = *seed + i4_huge;
        }
        r = ( float ) ( *seed ) * 4.656612875E-10;
        return r;
    }

    void save_grid(uint8_t *grid, size_t grid_size) {
        string filename =
            "grid_table_" + to_string(grid_size) + "x" + to_string(grid_size) + ".bin";

        cout << "Saving table " << grid_size << "x" << grid_size << " in file " << filename << endl;

        ofstream file(filename, ios::binary);
        if (!file) {
            cerr << "Failed to open file: " << filename << endl;
            return;
        }

        file.write(reinterpret_cast<const char *>(grid), grid_size * grid_size * sizeof(uint8_t)); // NOLINT
        file.close();
    }

    char* generate_rgb(int width, int height, uint8_t* grid, char* rgb) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int index = i * width + j;
                int color_index = grid[index];
                rgb[3 * index + 0] = STATE_COLORS[color_index][0];
                rgb[3 * index + 1] = STATE_COLORS[color_index][1];
                rgb[3 * index + 2] = STATE_COLORS[color_index][2];
            }
        }
        return rgb;
    }

    char* generate_rgb_packet(int width, int height, uint64_t* grid, char* rgb) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int cell_index = y * (ROW_SIZE) + (x / ELEMENTS_PER_CELL);
                uint64_t cell = grid[cell_index];
                int subcell_index = x % ELEMENTS_PER_CELL;
                uint8_t subcell = getSubCellH(cell, subcell_index);

                int rgb_index = 3 * (y * width + x);
                rgb[rgb_index + 0] = STATE_COLORS[subcell][0];
                rgb[rgb_index + 1] = STATE_COLORS[subcell][1];
                rgb[rgb_index + 2] = STATE_COLORS[subcell][2];
            }
        }
        return rgb;
    }

    void save_grid_to_png(uint8_t* X, int grid_size, int iteration) {
        int channels = 3;
        std::unique_ptr<uint8_t[]> image(new uint8_t[grid_size * grid_size * channels]);

        generate_rgb(grid_size, grid_size, reinterpret_cast<uint8_t*>(X), reinterpret_cast<char*>(image.get()));

        const fs::path output_dir = "output";
        fs::create_directories(output_dir);

        fs::path filename = output_dir / std::format("gol_{}.png", iteration);
        stbi_write_png(filename.string().c_str(), grid_size, grid_size, channels, image.get(), grid_size * channels);
    }
} // namespace utils
