#include "wireworld.hpp"
#include "kernels.hpp"

#include <iostream>
#include <utility>

using namespace std;

namespace {
    const int stride = 8; // Used to draw the initial wire world (conductor concentric squares);
    const float p_head = 0.0001;
    void generate_test_pattern(uint8_t *grid, size_t grid_size, int seed = time(nullptr)) {
        int i = 0, j = 0, k = 0;
        long int heads = 0;

        // Set grid to empty
        for (i = 0; i < grid_size; i++) {
            for (j = 0; j < grid_size; j++) {
                grid[i * (grid_size) + j] = EMPTY;
            }
        }

        // Set a "conductor cross" in the middle of the grid
        for (i = 0; i < grid_size; i++) {
            grid[i * (grid_size) + (grid_size / 2)] = CONDUCTOR;
        }
        for (j = 0; j < grid_size; j++) {
            grid[(grid_size / 2) * (grid_size) + j] = CONDUCTOR;
        }

        // Set concentric squares in the grid with STRIDE padding.
        for (k = stride; k < (grid_size) / 2; k = k + stride) {
            // Draw square:
            // Draw left side and right side:
            for (i = k * (grid_size) + k; i < (grid_size) * (grid_size)-k * (grid_size);
                 i += grid_size) {
                grid[i] = CONDUCTOR;
                grid[i + (grid_size)-k - k] = CONDUCTOR;
            }
            // Draw up side and down side:
            for (j = k * (grid_size) + k; j < k * (grid_size) + (grid_size)-k; j++) {
                grid[j] = CONDUCTOR;
            }
            for (j = (grid_size) * (grid_size)-k * (grid_size) + (grid_size)-k;
                 j >= (grid_size) * (grid_size)-k * (grid_size) + k; j--) {
                grid[j] = CONDUCTOR;
            }
        }

        // int seed = 0;
        // Set initial random heads
        for (i = 0; i < grid_size; i++) {
            for (j = 0; j < grid_size; j++) {
                if (grid[i * (grid_size) + j] == CONDUCTOR &&
                    utils::r4_uniform_01(&seed) < p_head) {
                    grid[i * (grid_size) + j] = HEAD;
                    heads++;
                }
            }
        }

        cout << "Generating test pattern; HEAD count = " << heads << endl;
    }
} // namespace

WireWorldCA::WireWorldCA() {
    grid = shared_ptr<uint8_t[]>(new uint8_t[GRID_SIZE * GRID_SIZE]);
    // generate_test_pattern(grid.get(), GRID_SIZE, 0);
    for (size_t i = 0; i < grid_size; i += 2) {
        for (size_t j = 0; j < grid_size; j++) {
            grid[i * (grid_size) + j] = CONDUCTOR;
            // heads++;
        }
        grid[i * (grid_size) + 0] = HEAD;
    }
    utils::save_grid_to_png_ww(grid.get(), GRID_SIZE, 9);
}
WireWorldCA::WireWorldCA(shared_ptr<uint8_t[]> grid) : grid(std::move(grid)) {}

// WireWorldCA::WireWorldCA(const string &filename) {}

void WireWorldCA::run(int iterations, int snapshot_interval) {
    auto grid1 = make_unique<uint8_t[]>(grid_size * grid_size);
    auto grid2 = make_unique<uint8_t[]>(grid_size * grid_size);

    // Copy the original grid into grid1
    copy(grid.get(), grid.get() + (grid_size * grid_size), grid1.get());
    utils::save_grid_to_png_ww(grid1.get(), grid_size, 0);

    for (int i = 0; i < iterations; i++) {
        kernels::wireworld::compute_next_gen(grid1.get(), grid2.get(), grid_size);
        if (i % snapshot_interval == 0) {
            utils::save_grid_to_png_ww(grid2.get(), grid_size, i);
        }
        swap(grid1, grid2);
    }

    // Copy final state back into original grid
    copy(grid1.get(), grid1.get() + (grid_size * grid_size), grid.get());
}
