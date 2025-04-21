#include "cyclic_ca.hpp"
#include "kernels.hpp"
#include <cstddef>
#include <cstdint>
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
    grid = shared_ptr<uint8_t[]>(new uint8_t[GRID_SIZE * GRID_SIZE]);
    // utils::generate_random_grid(reinterpret_cast<uint8_t*>(grid.get()), GRID_SIZE);
    test_grid1(grid.get(), GRID_SIZE);
}

CyclicCA::CyclicCA(shared_ptr<uint8_t[]> grid) : grid(grid) {}

// TODO: Change this function completely
uint8_t* CyclicCA::test_grid1(uint8_t* output_grid, int grid_size) {
    for(size_t i = 0; i < grid_size; ++i) {
        for (size_t j = 0; j < grid_size; ++j) {
            int index = i * grid_size + j;
            float randomize = utils::r4_uniform_01(&seed);
            if (randomize < P_STATE) {
                output_grid[index] = STATE1;
            } else if (randomize < P_STATE * 2) {
                output_grid[index] = STATE2;
            } else if (randomize < P_STATE * 3) {
                output_grid[index] = STATE3;
            } else if (randomize < P_STATE * 4) {
                output_grid[index] = STATE4;
            } else if (randomize < P_STATE * 5) {
                output_grid[index] = STATE5;
            } else if (randomize < P_STATE * 6) {
                output_grid[index] = STATE6;
            } else if (randomize < P_STATE * 7) {
                output_grid[index] = STATE7;
            } else if (randomize < P_STATE * 8) {
                output_grid[index] = STATE8;
            } else if (randomize < P_STATE * 9) {
                output_grid[index] = STATE9;
            } else if (randomize < P_STATE * 10) {
                output_grid[index] = STATE10;
            } else if (randomize < P_STATE * 11) {
                output_grid[index] = STATE11;
            } else if (randomize < P_STATE * 12) {
                output_grid[index] = STATE12;
            } else if (randomize < P_STATE * 13) {
                output_grid[index] = STATE13;
            } else if (randomize < P_STATE * 14) {
                output_grid[index] = STATE14;
            } else {
                output_grid[index] = STATE15;
            }
        }
    }
    return output_grid;
}

void CyclicCA::run(int iterations, int snapshotInterval) {
    // TODO: optimize if possible
    auto grid1 = make_unique<uint8_t[]>(grid_size * grid_size);
    auto grid2 = make_unique<uint8_t[]>(grid_size * grid_size);
    char *rgb= (char *)malloc (3 * sizeof(char) *(GRID_SIZE)*(GRID_SIZE));	

    // Copy the original grid into grid1
    copy(grid.get(), grid.get() + grid_size * grid_size, grid1.get());

    for (int i = 0; i < iterations; i++) {
        cyclic_compute_next_gen(grid1.get(), grid2.get(), grid_size);
        if (i % snapshotInterval == 0) {
            utils::save_grid_to_png(grid2.get(), grid_size, i);
        }
        swap(grid1, grid2);
    }

    // Copy final state back into original grid
    copy(grid1.get(), grid1.get() + grid_size * grid_size, grid.get());
}
