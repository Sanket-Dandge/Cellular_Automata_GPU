#pragma once

#include <cstddef>
#include <cstdint>

void compute_next_gen(bool *currentGrid, bool *nextGrid, size_t N);
// void cyclic_compute_next_gen(int *currentGrid, int *nextGrid, int N);
void cyclic_baseline(uint8_t *currentGrid, uint8_t *nextGrid, int N);
void cyclic_lookup_gen(uint8_t *currentGrid, uint8_t *nextGrid, int N);
void cyclic_packet_coding_gen(uint64_t *currentGrid, uint64_t *nextGrid, int N);
void setSubCellH(uint64_t* currentCell, char position, uint8_t subCell);
uint8_t getSubCellH(uint64_t currentCell, char position);
