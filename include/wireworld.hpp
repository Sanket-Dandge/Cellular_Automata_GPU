#pragma once

#include "utils.h"
#include <cstdint>
#include <string>
#include "common.hpp"

static const int default_grid_size = 256;

enum States : uint8_t {
    EMPTY = 0,
    HEAD = 17,
    TAIL = 1,
    CONDUCTOR = 2,
};

using namespace std;

class WireWorldCA {
  private:
    size_t grid_size = default_grid_size;

    void read_configuration_from_file(const string &filename);
    void load_grid_from_file(const string &filename);
    static pair<uint, uint> get_rle_size(const string &filename);

  public:
    shared_ptr<uint8_t[]> grid;

    WireWorldCA();
    WireWorldCA(shared_ptr<uint8_t[]> grid, uint grid_size);
    // WireWorldCA(const string &filename);
    // WireWorldCA(const AutomatonConfiguration &config);

    void run(int iterations, int snapshot_interval = 10, Implementation impl = BASE);
    [[nodiscard]] size_t get_grid_size() const { return grid_size; }
};
