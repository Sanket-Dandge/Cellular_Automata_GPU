#include "wireworld.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "cyclic_ca.hpp"
#include "game_of_life.hpp"
#include <chrono>
#include <cstdio>
#include <doctest.h>
#include <iostream>

// NOLINTBEGIN
TEST_CASE("Wireworld_base") {
    const int size = 128 * 128;
    shared_ptr<uint8_t[]> grid = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    shared_ptr<uint8_t[]> exp10 = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    for (int i = 0; i < size; i++) {
        grid[i] = 0;
        exp10[i] = 0;
    }
    auto ww = WireWorldCA(grid);
    ww.run(10, 1, BASE);

    CHECK(std::memcmp(grid.get(), exp10.get(), size) == 0);
}
// NOLINTEND
