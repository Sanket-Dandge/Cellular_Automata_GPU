#include "game_of_life.hpp"
#include "cyclic_ca.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>

struct ScopedTimer {
  private:
    string label;
    chrono::high_resolution_clock::time_point start;

  public:
    ScopedTimer(const string &lable) : label(lable), start(chrono::high_resolution_clock::now()) {}
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

    string config_file = argv[1];

    // auto config = AutomatonConfiguration(config_file);
    CyclicCA life;
    // utils::save_grid_to_png(life.grid.get(), life.get_grid_size(), 1);
    cout << life.get_grid_size() << endl;
    // for (int i = 1; i < 10; i++) {
    {int iters = 1000;
    ScopedTimer __t(format("Iterations-{}", iters));
    life.run(iters, 1);}
    // }

    return 0;
}
