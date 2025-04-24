#ifndef CYCLIC_CA_HPP
#define CYCLIC_CA_HPP

#include "utils.h"
#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>

#define GRID_SIZE 128

enum CellState {
    STATE1,
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
};

using namespace std;

class CyclicCA {
    private:
    int grid_size = GRID_SIZE;
    int seed = 1;

    void read_configuration_from_file(const string &filename);
    void load_grid_from_file(const string &filename);

    public:
        shared_ptr<uint8_t[]> grid;

        CyclicCA();
        CyclicCA(shared_ptr<uint8_t[]> grid);
        CyclicCA(const string& filename);

        void run(int iterations, int snapshot_interval = 10);
        static void create_lookup_table(uint8_t table[TOTAL_STATES][2]);
        uint8_t* test_grid1(uint8_t* grid, int size);
        [[nodiscard]] size_t get_grid_size() const { return grid_size; }
};

#endif // !CYCLIC_CA_HPP
