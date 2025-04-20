#pragma once

// TODO: add it to namespace
#include <cstddef>

void computeNextGen(bool *currentGrid, bool *nextGrid, int N);
void cyclicComputeNextGen(int *currentGrid, int *nextGrid, int N);
