#pragma once

// TODO: add it to namespace
#include <cstddef>
#include <cstdint>

void compute_next_gen(bool *currentGrid, bool *nextGrid, size_t N);
// void cyclic_compute_next_gen(int *currentGrid, int *nextGrid, int N);
void cyclic_compute_next_gen(uint8_t *currentGrid, uint8_t *nextGrid, int N);
