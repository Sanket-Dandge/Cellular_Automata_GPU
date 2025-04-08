#ifndef GAME_OF_LIFE_CUH
#define GAME_OF_LIFE_CUH

#include "utils.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

class GameOfLife {
  public:
    GameOfLife: left(-1.0f), right(1.0f), top(1.0f), bottom(-1.0f) {}
    GameOfLife(int argc, char** argv);
};

#endif
