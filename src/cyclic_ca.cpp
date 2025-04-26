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
    // test_grid_packet_coding1(packet_grid.get(), GRID_SIZE);
    test_grid1(grid.get(), GRID_SIZE);
}

void CyclicCA::create_lookup_table(uint8_t table[TOTAL_STATES][2]) {
    for (int i = 0; i < TOTAL_STATES; i++) {
        table[i][0] = i;            
        table[i][1] = (i + 1) % TOTAL_STATES;
    }
}

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

uint64_t* CyclicCA::test_grid_packet_coding1(uint64_t* output_grid, int size) {
    for (int i = 0; i < size; i++) {
        for(int j = 0; j < ROW_SIZE; j++) {
            for(int k = 0; k < ELEMENTS_PER_CELL; k++) {
                float randomize = utils::r4_uniform_01(&seed);
                if (randomize < P_STATE) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE1);
                } else if (randomize < P_STATE * 2) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE2);
                } else if (randomize < P_STATE * 3) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE3);
                } else if (randomize < P_STATE * 4) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE4);
                } else if (randomize < P_STATE * 5) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE5);
                } else if (randomize < P_STATE * 6) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE6);
                } else if (randomize < P_STATE * 7) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE7);
                } else if (randomize < P_STATE * 8) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE8);
                } else if (randomize < P_STATE * 9) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE9);
                } else if (randomize < P_STATE * 10) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE10);
                } else if (randomize < P_STATE * 11) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE11);
                } else if (randomize < P_STATE * 12) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE12);
                } else if (randomize < P_STATE * 13) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE13);
                } else if (randomize < P_STATE * 14) {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE14);
                } else {
                    setSubCellH(&output_grid[i * (ROW_SIZE) + j], k, STATE15);
                }

            }
        }
    }
    return output_grid;
}

void CyclicCA::run(int iterations, int snapshotInterval) {
    // Allocate two separate grids
    auto raw_grid1 = std::make_unique<uint8_t[]>(grid_size * grid_size * sizeof(uint64_t));
    auto raw_grid2 = std::make_unique<uint8_t[]>(grid_size * grid_size * sizeof(uint64_t));

    uint64_t* grid1 = reinterpret_cast<uint64_t*>(raw_grid1.get());
    uint64_t* grid2 = reinterpret_cast<uint64_t*>(raw_grid2.get());

    // Copy original grid into grid1
    std::copy(grid.get(),
              grid.get() + grid_size * grid_size,
              reinterpret_cast<uint8_t*>(grid1));

    for (int i = 0; i < iterations; i++) {
        cyclic_baseline(reinterpret_cast<uint8_t*>(grid1),
                        reinterpret_cast<uint8_t*>(grid2),
                        grid_size);
        // cyclic_lookup_gen(reinterpret_cast<uint8_t*>(grid1),
        //                 reinterpret_cast<uint8_t*>(grid2),
        //                 grid_size);

        // cyclic_packet_coding_gen(grid1, grid2, grid_size);

        if (i % snapshotInterval == 0) {
            utils::save_grid_to_png(reinterpret_cast<uint8_t*>(grid2), grid_size, i);
        }
        std::swap(grid1, grid2);
    }

    // Copy back to original grid
    std::copy(reinterpret_cast<uint8_t*>(grid1),
              reinterpret_cast<uint8_t*>(grid1) + grid_size * grid_size,
              grid.get());
}
