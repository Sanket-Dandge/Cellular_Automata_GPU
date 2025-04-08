#include "game_of_life.hpp"
#include "kernels.hpp"
#include "utils.h"

GameOfLife::GameOfLife() {
    grid = shared_ptr<bool[]>(new bool[GRID_SIZE * GRID_SIZE]);
    utils::generate_random_grid(grid.get(), GRID_SIZE);
}
GameOfLife::GameOfLife(shared_ptr<bool[]> grid) : grid(grid) {}
GameOfLife::GameOfLife(int argc, char **argv) {
    // TODO: Read config and load
}

void GameOfLife::run(int iterations, int snapshotInterval) {
    // TODO: optimize if possible
    auto grid1 = make_unique<bool[]>(gridSize * gridSize);
    auto grid2 = make_unique<bool[]>(gridSize * gridSize);

    // Copy the original grid into grid1
    copy(grid.get(), grid.get() + gridSize * gridSize, grid1.get());

    for (int i = 0; i < iterations; i++) {
        // TODO: Copy buffers to gpu and back
        computeNextGen(grid1.get(), grid2.get(), gridSize);
        if (i % snapshotInterval == 0) {
            utils::save_grid_to_png(grid2.get(), gridSize, i);
        }
        swap(grid1, grid2);
    }

    // Copy final state back into original grid
    copy(grid1.get(), grid1.get() + gridSize * gridSize, grid.get());
}
