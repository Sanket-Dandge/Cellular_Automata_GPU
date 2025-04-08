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

    void generate_random_grid(bool *grid, size_t grid_size) {
        cout << "Initializing random grid" << endl;
        srand(time(nullptr));
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

    void save_grid_to_png(const bool *grid, size_t grid_size, int iteration) {
        // Create output buffer
        unique_ptr<unsigned char[]> image(new unsigned char[grid_size * grid_size]);
        for (int i = 0; i < grid_size * grid_size; i++) {
            image[i] = grid[i] ? 255 : 0;
        }

        // Ensure output directory exists
        const fs::path output_dir = "output";
        fs::create_directories(output_dir); // No-op if it already exists

        // Construct full path: output/gol_<iteration>.png
        fs::path filename = output_dir / std::format("gol_{}.png", iteration);
        // Or use: fmt::format if you don't have C++20

        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        stbi_write_png(filename.string().c_str(), grid_size, grid_size, 1, image.get(), grid_size);
    }
} // namespace utils
