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
    const int ca_size = 128;
    const int size = ca_size * ca_size;
    shared_ptr<uint8_t[]> grid = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    shared_ptr<uint8_t[]> exp10 = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    for (int i = 0; i < size; i++) {
        grid[i] = EMPTY;
        exp10[i] = EMPTY;
    }
    for (int i = 1; i < 127; i++) {
        grid[(63 * ca_size) + i] = i == 1 ? HEAD : CONDUCTOR;
        exp10[(63 * ca_size) + i] = i == 11 ? HEAD : (i == 10 ? TAIL : CONDUCTOR);
    }
    auto ww = WireWorldCA(grid, ca_size);
    ww.run(10, 1, BASE);

    CHECK(std::memcmp(grid.get(), exp10.get(), size) == 0);
}

TEST_CASE("Wireworld_lut") {
    const int ca_size = 128;
    const int size = ca_size * ca_size;
    shared_ptr<uint8_t[]> grid = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    shared_ptr<uint8_t[]> exp10 = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    for (int i = 0; i < size; i++) {
        grid[i] = EMPTY;
        exp10[i] = EMPTY;
    }
    for (int i = 1; i < 127; i++) {
        grid[(63 * ca_size) + i] = i == 1 ? HEAD : CONDUCTOR;
        exp10[(63 * ca_size) + i] = i == 11 ? HEAD : (i == 10 ? TAIL : CONDUCTOR);
    }
    auto ww = WireWorldCA(grid, ca_size);
    ww.run(10, 1, LUT);

    CHECK(std::memcmp(grid.get(), exp10.get(), size) == 0);
}

TEST_CASE("Wireworld_packet_coding") {
    const int ca_size = 256;
    const int size = ca_size * ca_size;
    shared_ptr<uint8_t[]> grid = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    shared_ptr<uint8_t[]> exp10 = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
    for (int i = 0; i < size; i++) {
        grid[i] = EMPTY;
        exp10[i] = EMPTY;
    }
    for (int i = 1; i < 127; i++) {
        grid[(63 * ca_size) + i] = i == 1 ? HEAD : CONDUCTOR;
        exp10[(63 * ca_size) + i] = i == 11 ? HEAD : (i == 10 ? TAIL : CONDUCTOR);
    }
    auto ww = WireWorldCA(grid, ca_size);
    ww.run(10, 1, PACKET_CODING);

    CHECK(std::memcmp(grid.get(), exp10.get(), size) == 0);
}
// NOLINTEND
