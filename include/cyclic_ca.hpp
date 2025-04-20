#ifndef CYCLIC_CA_HPP
#define CYCLIC_CA_HPP

#include "utils.h"
#include <filesystem>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>

#define GRID_SIZE 1024

typedef enum {
    STATE1 = 1,
    STATE2,
    STATE3,
    STATE4,
    STATE5,
    STATE6,
    STATE7,
    STATE8,
    STATE9,
    STATE10,
    STATE11,
    STATE12,
    STATE13,
    STATE14,
    STATE15,
} CellState;

using namespace std;
namespace fs = std::filesystem;

class AutomateConfiguration {
  private:
    unordered_map<string, string> parse(const fs::path &filename);

  public:
    optional<fs::path> gridFile;
    bool generateRandom;
    string size;
    int generations;

    AutomateConfiguration(const fs::path &filename);
};

class CyclicCA {
    private:
    int gridSize = GRID_SIZE;

    void read_configuration_from_file(const string &filename);
    void load_grid_from_file(const string &filename);
    static pair<uint, uint> get_rle_size(const string &filename);

    public:
        shared_ptr<int[]> grid;

        CyclicCA();
        CyclicCA(shared_ptr<int[]> grid);
        CyclicCA(const string& filename);
        CyclicCA(const AutomateConfiguration& config);

        void run(int iterations, int snapShotInterval = 10);
        int getGridSize() { return gridSize; }
};

#endif // !CYCLIC_CA_HPP
