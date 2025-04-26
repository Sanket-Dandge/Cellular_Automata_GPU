#pragma once

#include "utils.h"
#include <cstdint>
#include <string>

#define GRID_SIZE 1024

enum States : uint8_t {
    EMPTY = 0,
    HEAD = 1,
    TAIL = 2,
    CONDUCTOR = 3,
};

using namespace std;

class WireWorldCA {
  private:
    size_t grid_size = GRID_SIZE;

    void read_configuration_from_file(const string &filename);
    void load_grid_from_file(const string &filename);
    static pair<uint, uint> get_rle_size(const string &filename);

  public:
    shared_ptr<uint8_t[]> grid;

    WireWorldCA();
    WireWorldCA(shared_ptr<uint8_t[]> grid);
    // WireWorldCA(const string &filename);
    // WireWorldCA(const AutomatonConfiguration &config);

    void run(int iterations, int snapshot_interval = 10);
    [[nodiscard]] size_t get_grid_size() const { return grid_size; }
};
