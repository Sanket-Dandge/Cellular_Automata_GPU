#ifndef GAME_OF_LIFE_CUH
#define GAME_OF_LIFE_CUH

#include "utils.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>

#define GRID_SIZE 1024

using namespace std;
namespace fs = std::filesystem;

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
struct AutomatonConfiguration {
  private:
    static unordered_map<string, string> parse(const fs::path &filename);

  public:
    optional<fs::path> grid_file;
    bool generate_random;
    string size;
    int generations;

    AutomatonConfiguration(const fs::path &filename);
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

class GameOfLife {
  private:
    size_t grid_size = GRID_SIZE;

    void read_configuration_from_file(const string &filename);
    void load_grid_from_file(const string &filename);
    static pair<uint, uint> get_rle_size(const string &filename);

  public:
    shared_ptr<uint8_t[]> grid;

    GameOfLife();
    GameOfLife(shared_ptr<uint8_t[]> grid);
    GameOfLife(const string &filename);
    GameOfLife(const AutomatonConfiguration &config);

    void run(int iterations, int snapshot_interval = 10);
    static void save_gol_grid_to_png(const uint8_t *grid, uint grid_size, int iteration);
    [[nodiscard]] size_t get_grid_size() const { return grid_size; }
};

#endif
