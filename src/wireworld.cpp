#include "wireworld.hpp"
#include "kernels.hpp"

#include <utility>

WireWorldCA::WireWorldCA() {
    grid = shared_ptr<uint8_t[]>(new uint8_t[GRID_SIZE * GRID_SIZE]);
    utils::generate_random_grid(grid.get(), GRID_SIZE, 0, 4);
}
WireWorldCA::WireWorldCA(shared_ptr<uint8_t[]> grid) : grid(std::move(grid)) {}

// WireWorldCA::WireWorldCA(const string &filename) {}

void WireWorldCA::run(int iterations, int snapshot_interval) {
    auto grid1 = make_unique<uint8_t[]>(grid_size * grid_size);
    auto grid2 = make_unique<uint8_t[]>(grid_size * grid_size);

    // Copy the original grid into grid1
    copy(grid.get(), grid.get() + (grid_size * grid_size), grid1.get());

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
