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
Run different automatons
Usage:
  program [OPTION...]

  -b, --benchmark              Run in benchmark mode
  -s, --snapshot-interval arg  Number of frames to take snapshot after,
                               (ignored in benchmark mode) (default: 10)
  -a, --automaton arg          Atomaton to run (gol/cca/ww/ff) (default: gol)
  -c, --config arg             Config file to read (only for gol) (default: )
  -k, --kernel arg             Kernel to use (base/lut/pc) (default: base)
  -g, --generations arg        Number of generations to run simulation for
                               (default: 1024)
  -h, --help                   Print usage
```
