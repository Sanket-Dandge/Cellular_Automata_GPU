#include "game_of_life.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <filesystem>

#include "stb_image_write.h"
#include <bit>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

void GameOfLife::save_gol_grid_to_png(const uint8_t *grid, uint grid_size, int iteration) {
    int channels = 1;
    std::unique_ptr<uint8_t[]> image(new uint8_t[grid_size * grid_size * channels]);

    for (uint i = 0; i < grid_size; i++) {
        for (uint j = 0; j < grid_size; j++) {
            if (grid[i * grid_size + j]) {
                image[i * grid_size + j] = 255;
            } else {
                image[i * grid_size + j] = 0;
            }
        }
    }

    const fs::path output_dir = "output";
    fs::create_directories(output_dir);

    fs::path filename = output_dir / std::format("gol_{}.png", iteration);
    stbi_write_png(filename.string().c_str(), grid_size, grid_size, channels, image.get(),
                   grid_size * channels);
}

GameOfLife::GameOfLife() {
    grid = shared_ptr<uint8_t[]>(new uint8_t[GRID_SIZE * GRID_SIZE]);
    utils::generate_random_grid(grid.get(), GRID_SIZE);
}
GameOfLife::GameOfLife(shared_ptr<uint8_t[]> grid) : grid(std::move(grid)) {}

GameOfLife::GameOfLife(const AutomatonConfiguration &config) {
    if (config.size == "nextPower") {
        if (config.generate_random) {
            cerr << "cannot use nextPower with random" << endl;
            exit(1);
        } else {
            auto pattern_size = get_rle_size(config.grid_file->string());
            grid_size = bit_ceil(max({pattern_size.first, pattern_size.second, 32U}));
        }
    } else {
        grid_size = stoi(config.size);
    }

    grid = shared_ptr<uint8_t[]>(new uint8_t[grid_size * grid_size]);

    if (config.generate_random) {
        utils::generate_random_grid(grid.get(), grid_size);
    } else {
        load_grid_from_file(config.grid_file->string());
    }
}
GameOfLife::GameOfLife(const string &filename) : GameOfLife(AutomatonConfiguration(filename)) {}

void GameOfLife::run(int iterations, int snapshot_interval, Implementation impl) {
    switch (impl) {
    case PACKET_CODING: {
        for (int i = 0; i < iterations;) {
            kernels::gol::compute_next_gen_packet_coding(grid.get(), grid_size, snapshot_interval);
            i += snapshot_interval;
            save_gol_grid_to_png(grid.get(), grid_size, i);
        }
        break;
    }
    default: {
        for (int i = 0; i < iterations;) {
            kernels::gol::compute_next_gen(grid.get(), grid_size, snapshot_interval);
            i += snapshot_interval;
            save_gol_grid_to_png(grid.get(), grid_size, i);
        }
        break;
    }
    }
}

AutomatonConfiguration::AutomatonConfiguration(const fs::path &filename) {
    auto config = parse(filename);

    // Resolve gridFile path
    if (config.contains("gridFile") && config["gridFile"] != "random") {
        grid_file = fs::path(config["gridFile"]);

        // Make it relative to config file location if it's not absolute
        if (grid_file && !grid_file->is_absolute()) {
            grid_file = filename.parent_path() / *grid_file;
        }
        generate_random = false;
    } else {
        grid_file = nullopt;
        generate_random = true;
    }
    size = config.contains("size") ? config["size"] : "nextPower";
    string generations_config = config.contains("generations") ? config["generations"] : "1000";
    generations = stoi(generations_config);

    const unordered_set<string> allowed_keys = {"gridFile", "size", "generations"};
    for (auto kv_pair : config) {
        if (!allowed_keys.contains(kv_pair.first)) {
            cerr << "Unknown key found in configuration `" << kv_pair.first << "'";
        }
    }
}

unordered_map<string, string> AutomatonConfiguration::parse(const fs::path &filename) {
    unordered_map<string, string> config;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines or comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Split key and value
        istringstream iss(line);
        string key, value;
        iss >> key >> value;
        if (!key.empty() && !value.empty()) {
            config[key] = value;
        }
    }

    return config;
}

pair<uint, uint> GameOfLife::get_rle_size(const string &filename) {
    if (!fs::exists(filename)) {
        throw runtime_error("File does not exist: " + filename);
    }

    ifstream file(filename);
    if (!file) {
        throw runtime_error("Failed to open file: " + filename);
    }

    string line;
    size_t pattern_width = 0, pattern_height = 0;

    // Parse header and RLE data
    while (getline(file, line)) {
        // Skip comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line.starts_with("x =")) {
            // Header line: parse dimensions
            smatch match;
            regex header_regex(R"(x\s*=\s*(\d+),\s*y\s*=\s*(\d+))");
            if (regex_search(line, match, header_regex)) {
                pattern_width = stoi(match[1]);
                pattern_height = stoi(match[2]);
                return {pattern_width, pattern_height};
            }
        }
    }
    return {GRID_SIZE, GRID_SIZE};
}

// NOLINTBEGIN
void GameOfLife::load_grid_from_file(const string &filename) {
    cout << "Reading initial grid setup from " << filename << endl;
    if (!fs::exists(filename)) {
        throw runtime_error("File does not exist: " + filename);
    }

    ifstream file(filename);
    if (!file) {
        throw runtime_error("Failed to open file: " + filename);
    }

    // Clear the grid
    for (size_t i = 0; i < grid_size * grid_size; ++i) {
        grid[i] = false;
    }

    string line;
    string rle_data;
    size_t pattern_width = 0, pattern_height = 0;

    // Parse header and RLE data
    while (getline(file, line)) {
        // Skip comments
        if (line.empty() || line[0] == '#')
            continue;

        if (line.rfind("x =", 0) == 0) {
            // Header line: parse dimensions
            smatch match;
            regex header_regex(R"(x\s*=\s*(\d+),\s*y\s*=\s*(\d+))");
            if (regex_search(line, match, header_regex)) {
                pattern_width = stoi(match[1]);
                pattern_height = stoi(match[2]);
                if (pattern_width > grid_size || pattern_height > grid_size) {
                    throw runtime_error("Pattern too big for current grid");
                }
            }
        } else {
            rle_data += line;
        }
    }

    // Parse RLE body
    size_t x = 0, y = 0;
    size_t run_count = 0;

    for (size_t i = 0; i < rle_data.size(); ++i) {
        char c = rle_data[i];

        if (isdigit(c)) {
            run_count = run_count * 10 + (c - '0');
        } else if (c == 'b' || c == 'o') {
            size_t count = run_count ? run_count : 1;
            uint8_t value = (c == 'o');

            for (size_t j = 0; j < count; ++j) {
                if (x < grid_size && y < grid_size)
                    grid[y * grid_size + x] = value;
                x++;
            }

            run_count = 0;
        } else if (c == '$') {
            size_t count = run_count ? run_count : 1;
            y += count;
            x = 0;
            run_count = 0;
        } else if (c == '!') {
            break;
        }
    }
}
// NOLINTEND
