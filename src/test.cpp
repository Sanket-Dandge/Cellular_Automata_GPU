#include "game_of_life.hpp"
#include <iostream>

int main() {
    GameOfLife life("testconfig.txt");
    cout << "Grid size = " << life.getGridSize() << endl;
    utils::save_grid_to_png(life.grid.get(), life.getGridSize(), 1000);
    life.run(1000, 10);
}
