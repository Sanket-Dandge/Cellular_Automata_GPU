#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

void compute_next_gen(bool *currentGrid, bool *nextGrid, size_t N);
// void cyclic_compute_next_gen(int *currentGrid, int *nextGrid, int N);
void cyclic_baseline(uint8_t *currentGrid, uint8_t *nextGrid, int N);
void cyclic_lookup_gen(uint8_t *currentGrid, uint8_t *nextGrid, int N);
void cyclic_packet_coding_gen(uint64_t *currentGrid, uint64_t *nextGrid, int N);
void setSubCellH(uint64_t *currentCell, char position, uint8_t subCell);
uint8_t getSubCellH(uint64_t currentCell, char position);

namespace kernels::wireworld {
    void compute_next_gen_base(uint8_t *current_grid, uint ca_grid_size, size_t niter);
    void compute_next_gen_lut(uint8_t *current_grid, uint ca_grid_size, size_t niter);
    void compute_next_gen_packet_coding(uint8_t *current_grid, uint ca_grid_size, size_t niter);
}
namespace kernels::gol {
    void compute_next_gen(uint8_t *current_grid, size_t ca_grid_size, size_t niter);
}
