# Automaton Project

## 🛠️ Build Instructions

Make sure you have `cmake`, a C++ compiler (e.g. `g++`, `clang++`) and a CUDA compiler(e.g. `nvcc`).

```bash
# Create a build directory
mkdir build
cd build

# Generate build files with CMake
cmake ..

# Compile the project
make
```

## 🚀 Run

After building, run the executable with a configuration file:

```bash
./bin/game_of_life.o tests/test1.txt
```
