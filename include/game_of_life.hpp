#ifndef GAME_OF_LIFE_CUH
#define GAME_OF_LIFE_CUH

#include "utils.h"
#include <memory>

#define GRID_SIZE 1024

using namespace std;

class GameOfLife {
  private:
    shared_ptr<bool[]> grid;
    int gridSize = GRID_SIZE;

  public:
    GameOfLife();
    GameOfLife(shared_ptr<bool[]> grid);
    GameOfLife(int argc, char **argv);

    void run(int iterations, int snapshotInterval = 10);
};

#endif
