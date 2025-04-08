#include "game_of_life.cuh"
#include "kernels.cuh"

GameOfLife::GameOfLife(int argc, char** argv) :
  windowWidth_(800), windowHeight_(800), left(-1.0f), right(1.0f), top(1.0f),
  bottom(-1.0f), gpuOn_(true), gpuMethod_(1), cellPerThread_(4)
{

}
