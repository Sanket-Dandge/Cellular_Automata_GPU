#ifndef GAME_OF_LIFE_CUH
#define GAME_OF_LIFE_CUH

#include "utils.h"
#include <filesystem>
#include <memory>
#include <unordered_map>

#define GRID_SIZE 1024

using namespace std;
namespace fs = std::filesystem;

class AutomatonConfiguration {
  private:
    unordered_map<string, string> parse(const fs::path &filename);

  public:
    string gridFile;
    string size;
    int generations;

    AutomatonConfiguration(const fs::path &filename);
};

class GameOfLife {
  private:
    int gridSize = GRID_SIZE;

    void read_configuration_from_file(const string &filename);
    void load_grid_from_file(const string &filename);
    static pair<uint, uint> get_rle_size(const string &filename);

  public:
    shared_ptr<bool[]> grid;

    GameOfLife();
    GameOfLife(shared_ptr<bool[]> grid);
    GameOfLife(const string &filename);
    GameOfLife(const AutomatonConfiguration &config);

    void run(int iterations, int snapshotInterval = 10);
    int getGridSize() { return gridSize; }
};

#endif
