#include "game_of_life.hpp"
#include <chrono>
#include <iostream>

struct ScopedTimer {
    string label;
    chrono::high_resolution_clock::time_point start;

    ScopedTimer(const string &l) : label(l), start(chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = chrono::high_resolution_clock::now();
        auto dur = chrono::duration_cast<chrono::microseconds>(end - start).count();
        cout << label << ": " << dur << " μs" << endl;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file>" << endl;
        return 1;
    }

    string configFile = argv[1];

    auto config = AutomatonConfiguration(configFile);
    GameOfLife life(config);
    utils::save_grid_to_png(life.grid.get(), life.getGridSize(), 1);

    for (int i = 1; i < 10; i++) {
        int iters = 2u << i;
        ScopedTimer __t(format("Iterations-{}", iters));
        life.run(iters, 1);
    }

    return 0;
}
