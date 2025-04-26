#include "game_of_life.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <filesystem>

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

void GameOfLife::run(int iterations, int snapshot_interval) {
    // TODO: optimize if possible
    auto grid1 = make_unique<uint8_t[]>(grid_size * grid_size);
    auto grid2 = make_unique<uint8_t[]>(grid_size * grid_size);

    // Copy the original grid into grid1
    copy(grid.get(), grid.get() + (grid_size * grid_size), grid1.get());

    for (int i = 0; i < iterations; i++) {
        kernels::gol::compute_next_gen(grid1.get(), grid2.get(), grid_size, 1);
        if (i % snapshot_interval == 0) {
            utils::save_grid_to_png(grid2.get(), grid_size, i);
        }
        swap(grid1, grid2);
    }

    // Copy final state back into original grid
    copy(grid1.get(), grid1.get() + (grid_size * grid_size), grid.get());
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
