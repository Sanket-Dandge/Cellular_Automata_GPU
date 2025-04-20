#include "cyclic_ca.hpp"
#include "kernels.hpp"
#include <filesystem>

#include <bit>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;


CyclicCA::CyclicCA() {
    grid = shared_ptr<int[]>(new int[GRID_SIZE * GRID_SIZE]);
    utils::generate_random_grid(grid.get(), GRID_SIZE);
}

CyclicCA::CyclicCA(shared_ptr<int[]> grid) : grid(grid) {}

CyclicCA::CyclicCA(const AutomateConfiguration& config) {
    if (config.size == "nextPower") {
        if (config.generateRandom) {
            cerr << "cannot use nextPower with random" << endl;
            exit(1);
        } else {
            auto patternSize = get_rle_size(config.gridFile->string());
            gridSize = bit_ceil(max(max(patternSize.first, patternSize.second), 32u));
        }
    } else {
        gridSize = stoi(config.size);
    }

    grid = shared_ptr<int[]>(new int[gridSize * gridSize]);

    if (config.generateRandom) {
        utils::generate_random_grid(grid.get(), gridSize);
    } else {
        load_grid_from_file(config.gridFile->string());
    }
}

CyclicCA::CyclicCA(const string& filename) : CyclicCA(AutomateConfiguration(filename)) {}

void CyclicCA::run(int iterations, int snapshotInterval) {
    // TODO: optimize if possible
    auto grid1 = make_unique<int[]>(gridSize * gridSize);
    auto grid2 = make_unique<int[]>(gridSize * gridSize);

    // Copy the original grid into grid1
    copy(grid.get(), grid.get() + gridSize * gridSize, grid1.get());

    for (int i = 0; i < iterations; i++) {
        cyclicComputeNextGen(grid1.get(), grid2.get(), gridSize);
        if (i % snapshotInterval == 0) {
            utils::save_grid_to_png(grid2.get(), gridSize, i);
        }
        swap(grid1, grid2);
    }

    // Copy final state back into original grid
    copy(grid1.get(), grid1.get() + gridSize * gridSize, grid.get());
}

pair<uint, uint> CyclicCA::get_rle_size(const string &filename) {
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
        if (line.empty() || line[0] == '#')
            continue;

        if (line.rfind("x =", 0) == 0) {
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

void CyclicCA::load_grid_from_file(const string &filename) {
    cout << "Reading initial grid setup from " << filename << endl;
    if (!fs::exists(filename)) {
        throw runtime_error("File does not exist: " + filename);
    }

    ifstream file(filename);
    if (!file) {
        throw runtime_error("Failed to open file: " + filename);
    }

    // Clear the grid
    for (size_t i = 0; i < gridSize * gridSize; ++i) {
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
                if (pattern_width > gridSize || pattern_height > gridSize) {
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
            bool value = (c == 'o');

            for (size_t j = 0; j < count; ++j) {
                if (x < gridSize && y < gridSize)
                    grid[y * gridSize + x] = value;
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

AutomateConfiguration::AutomateConfiguration(const fs::path &filename) {
    auto config = parse(filename);

    // Resolve gridFile path
    if (config.contains("gridFile") && config["gridFile"] != "random") {
        gridFile = fs::path(config["gridFile"]);

        // Make it relative to config file location if it's not absolute
        if (gridFile && !gridFile->is_absolute()) {
            gridFile = filename.parent_path() / *gridFile;
        }
    } else {
        gridFile = nullopt;
        generateRandom = true;
    }
    size = config.contains("size") ? config["size"] : "nextPower";
    string generationsConfig = config.contains("generations") ? config["generations"] : "1000";
    generations = stoi(generationsConfig);

    const unordered_set<string> allowedKeys = {"gridFile", "size", "generations"};
    for (auto kv : config) {
        if (!allowedKeys.contains(kv.first)) {
            cerr << "Unknown key found in configuration `" << kv.first << "'";
        }
    }
}

unordered_map<string, string> AutomateConfiguration::parse(const fs::path &filename) {
    unordered_map<string, string> config;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines or comments
        if (line.empty() || line[0] == '#')
            continue;

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
